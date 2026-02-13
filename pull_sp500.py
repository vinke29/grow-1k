import json
import csv
import time
import urllib.request
import re
import os
import sys

API_KEY = "JNGL6W6EOSQXDK01"
BASE_URL = "https://www.alphavantage.co/query"
DIR = os.path.dirname(os.path.abspath(__file__))

QUARTERS = [
    "2023Q1", "2023Q2", "2023Q3", "2023Q4",
    "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    "2025Q1", "2025Q2", "2025Q3", "2025Q4",
]

# --- AI detection ---
AI_KEYWORDS = [
    "artificial intelligence", " ai ", " ai,", " ai.", " ai;", " ai:", " ai'",
    " ai-", "machine learning", "generative ai", "generative artificial",
    "gen ai", "genai", "large language model", " llm",
    "deep learning", "neural network", "copilot", "chatgpt", "openai",
    "natural language processing", "computer vision",
    "apple intelligence", "gemini", "claude",
]

def has_ai_mention(text):
    lower = " " + text.lower() + " "
    return any(kw in lower for kw in AI_KEYWORDS)

# --- Categorization (by AI use case) ---
CATEGORY_RULES = {
    "AI-Native Product": [
        "our ai product", "ai platform", "ai service", "copilot", "ai offering",
        "ai-powered product", "ai solution we", "our generative ai",
        "gpu", "accelerat", "inference", "training workload",
    ],
    "AI-Enhanced Feature": [
        "added ai", "ai feature", "ai-powered", "ai-driven", "integrate ai",
        "leveraging ai", "using ai to", "ai capabilities in our",
        "recommendation", "personalization", "ai search", "ai assistant",
    ],
    "Internal Automation": [
        "internal", "automat", "productivity", "efficiency", "workflow",
        "coding assistant", "developer productivity", "customer service",
        "call center", "back office", "ai tool", "employee",
    ],
    "AI Infrastructure": [
        "data center", "cloud", "compute", "infrastructure", "hyperscal",
        "capital expenditure", "capex", "server", "chip", "semiconductor",
        "power", "energy demand", "cooling",
    ],
    "Risk/Regulatory": [
        "regulat", "responsible ai", "ai governance", "ethical",
        "compliance", "ai risk", "ai safety", "bias", "privacy",
        "legislation", "oversight",
    ],
}

def categorize(text):
    lower = text.lower()
    matched = []
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in lower for kw in keywords):
            matched.append(category)
    if not matched:
        # Check if it's vague/buzzword
        vague = ["excited about ai", "opportunity", "exploring", "potential of ai",
                 "believe in ai", "ai is important", "ai journey"]
        if any(v in lower for v in vague):
            return ["Vague/Buzzword"]
        return ["Uncategorized"]
    return matched

# --- Who brought it up ---
def classify_speaker(title):
    if not title:
        return "Unknown"
    lower = title.lower()
    analyst_terms = ["analyst", "research", "managing director", "equity"]
    if any(t in lower for t in analyst_terms):
        return "Analyst"
    return "Executive"

# --- Quantified impact ---
def has_quantified_impact(text):
    patterns = [
        r'\$[\d,.]+\s*(million|billion|m\b|b\b)',
        r'\d+\s*%',
        r'\d+x\b',
        r'doubled|tripled|quadrupled',
        r'\d+\s*(million|billion)\s*(dollar|user|customer|parameter)',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

# --- Best quote extraction ---
def extract_best_quote(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Prefer sentences with both AI mention and quantified impact
    for s in sentences:
        if has_ai_mention(s) and has_quantified_impact(s):
            return s.strip()[:300]
    # Then sentences with AI mention
    for s in sentences:
        if has_ai_mention(s):
            return s.strip()[:300]
    return text[:300]

# --- API fetch ---
def fetch_transcript(symbol, quarter):
    url = f"{BASE_URL}?function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={quarter}&apikey={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "grow-1k-research"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return None

# --- Main ---
def main():
    with open(os.path.join(DIR, "sp500.json")) as f:
        companies = json.load(f)

    print(f"Pulling transcripts for {len(companies)} companies x {len(QUARTERS)} quarters = {len(companies) * len(QUARTERS)} calls")
    print(f"Rate: ~75/min, estimated time: ~{len(companies) * len(QUARTERS) // 75} minutes\n")

    summary_rows = []
    quote_rows = []
    calls_made = 0
    calls_this_minute = 0
    minute_start = time.time()

    # Resume support: check if we have partial results
    progress_file = os.path.join(DIR, "progress.json")
    done_keys = set()
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            done_keys = set(json.load(f))
        print(f"Resuming: {len(done_keys)} already done\n")

    for ci, company in enumerate(companies):
        symbol = company["symbol"]
        sector = company["sector"]

        for quarter in QUARTERS:
            key = f"{symbol}_{quarter}"
            if key in done_keys:
                continue

            # Rate limiting: max 70/min to be safe
            calls_this_minute += 1
            if calls_this_minute >= 70:
                elapsed = time.time() - minute_start
                if elapsed < 62:
                    wait = 62 - elapsed
                    print(f"  Rate limit pause: {wait:.0f}s")
                    time.sleep(wait)
                calls_this_minute = 0
                minute_start = time.time()

            data = fetch_transcript(symbol, quarter)
            calls_made += 1

            if not data or "transcript" not in data or not data["transcript"]:
                done_keys.add(key)
                continue

            transcript = data["transcript"]
            total_segments = len(transcript)
            ai_segments = [seg for seg in transcript if has_ai_mention(seg["content"])]
            ai_count = len(ai_segments)

            if ai_count == 0:
                summary_rows.append({
                    "Company": symbol,
                    "Sector": sector,
                    "Quarter": quarter,
                    "AI Mentions": 0,
                    "Total Segments": total_segments,
                    "AI Intensity %": "0.0",
                    "Top Category": "",
                    "Who Brought It Up": "",
                    "Quantified Impact": "No",
                    "Best Quote": "",
                    "Speaker": "",
                })
                done_keys.add(key)
                continue

            # Analyze AI segments
            all_categories = {}
            exec_mentions = 0
            analyst_mentions = 0
            any_quantified = False
            best_quote = ""
            best_speaker = ""
            best_sentiment = None

            for seg in ai_segments:
                cats = categorize(seg["content"])
                for c in cats:
                    all_categories[c] = all_categories.get(c, 0) + 1

                role = classify_speaker(seg.get("title", ""))
                if role == "Analyst":
                    analyst_mentions += 1
                else:
                    exec_mentions += 1

                if has_quantified_impact(seg["content"]):
                    any_quantified = True

                sentiment = seg.get("sentiment", "")

                # Add to quotes CSV
                quote_rows.append({
                    "Company": symbol,
                    "Sector": sector,
                    "Quarter": quarter,
                    "Speaker": seg.get("speaker", ""),
                    "Title": seg.get("title", ""),
                    "Role": role,
                    "Categories": ", ".join(cats),
                    "Quantified": "Yes" if has_quantified_impact(seg["content"]) else "No",
                    "Sentiment": sentiment,
                    "Quote": extract_best_quote(seg["content"]),
                })

            top_category = max(all_categories, key=all_categories.get) if all_categories else ""
            who = "Executive" if exec_mentions >= analyst_mentions else "Analyst"
            if exec_mentions > 0 and analyst_mentions > 0:
                who = f"Both ({exec_mentions}E/{analyst_mentions}A)"
            intensity = round(ai_count / total_segments * 100, 1)

            # Best quote for summary: prefer exec with quantified impact
            for seg in ai_segments:
                if has_quantified_impact(seg["content"]) and classify_speaker(seg.get("title", "")) == "Executive":
                    best_quote = extract_best_quote(seg["content"])
                    best_speaker = seg.get("speaker", "")
                    break
            if not best_quote:
                for seg in ai_segments:
                    if classify_speaker(seg.get("title", "")) == "Executive":
                        best_quote = extract_best_quote(seg["content"])
                        best_speaker = seg.get("speaker", "")
                        break
            if not best_quote:
                best_quote = extract_best_quote(ai_segments[0]["content"])
                best_speaker = ai_segments[0].get("speaker", "")

            summary_rows.append({
                "Company": symbol,
                "Sector": sector,
                "Quarter": quarter,
                "AI Mentions": ai_count,
                "Total Segments": total_segments,
                "AI Intensity %": str(intensity),
                "Top Category": top_category,
                "Who Brought It Up": who,
                "Quantified Impact": "Yes" if any_quantified else "No",
                "Best Quote": best_quote,
                "Speaker": best_speaker,
            })

            done_keys.add(key)

            # Save progress every 50 calls
            if calls_made % 50 == 0:
                with open(progress_file, "w") as f:
                    json.dump(list(done_keys), f)
                # Write partial CSVs
                write_csvs(summary_rows, quote_rows)
                print(f"  [{calls_made} calls] {symbol} {quarter} — {ai_count} AI mentions")

        # Print progress per company
        if calls_made % 100 < len(QUARTERS):
            pct = (ci + 1) / len(companies) * 100
            print(f"[{pct:.0f}%] {symbol} done ({calls_made} calls total)")

    # Final save
    with open(progress_file, "w") as f:
        json.dump(list(done_keys), f)
    write_csvs(summary_rows, quote_rows)

    print(f"\nDone! {calls_made} API calls made.")
    print(f"Summary: {len(summary_rows)} rows -> ai_summary.csv")
    print(f"Quotes:  {len(quote_rows)} rows -> ai_quotes.csv")


def write_csvs(summary_rows, quote_rows):
    summary_path = os.path.join(DIR, "ai_summary.csv")
    summary_fields = ["Company", "Sector", "Quarter", "AI Mentions", "Total Segments",
                       "AI Intensity %", "Top Category", "Who Brought It Up",
                       "Quantified Impact", "Best Quote", "Speaker"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    quotes_path = os.path.join(DIR, "ai_quotes.csv")
    quotes_fields = ["Company", "Sector", "Quarter", "Speaker", "Title", "Role",
                      "Categories", "Quantified", "Sentiment", "Quote"]
    with open(quotes_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=quotes_fields)
        w.writeheader()
        w.writerows(quote_rows)


if __name__ == "__main__":
    main()
