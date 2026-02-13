import json
import csv
import time
import urllib.request
import re
import os

API_KEY = "D3M5XDTIFT5AC8V4"
BASE_URL = "https://www.alphavantage.co/query"

COMPANIES = [
    ("MSFT", "Big Tech"),
    ("NVDA", "Semiconductors"),
    ("JPM", "Finance"),
    ("UNH", "Healthcare"),
    ("WMT", "Retail"),
    ("CAT", "Industrial"),
    ("PFE", "Pharma"),
    ("NEE", "Energy"),
]

QUARTERS = ["2023Q4", "2025Q4"]

AI_KEYWORDS = [
    "artificial intelligence", " ai ", " ai,", " ai.", " ai;", " ai:", " ai'",
    "machine learning", "generative", "large language model", "llm",
    "deep learning", "neural network", "copilot", "chatgpt", "openai",
    "automation", "predictive analytics", "natural language processing",
    "computer vision",
]

CATEGORIES = {
    "Product": ["product", "feature", "customer-facing", "user experience", "copilot", "launch", "release", "platform", "offering", "service"],
    "Operations": ["operations", "efficiency", "productivity", "internal", "workflow", "process", "supply chain", "logistics"],
    "Cost Reduction": ["cost", "save", "saving", "reduce", "reduction", "margin", "expense", "optimize"],
    "Revenue Growth": ["revenue", "growth", "sales", "monetize", "demand", "opportunity", "market share", "commercial"],
    "R&D": ["research", "develop", "r&d", "invest", "innovation", "build", "train", "model", "infrastructure", "data center", "gpu", "chip"],
    "Risk/Regulatory": ["risk", "regulation", "compliance", "responsible", "ethical", "governance", "security", "privacy", "threat"],
}


def fetch_transcript(symbol, quarter):
    url = f"{BASE_URL}?function=EARNINGS_CALL_TRANSCRIPT&symbol={symbol}&quarter={quarter}&apikey={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "grow-1k-research"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching {symbol} {quarter}: {e}")
        return None


def has_ai_mention(text):
    lower = text.lower()
    return any(kw in lower for kw in AI_KEYWORDS)


def categorize(text):
    lower = text.lower()
    matched = []
    for category, keywords in CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            matched.append(category)
    return matched if matched else ["General"]


def has_quantified_impact(text):
    patterns = [
        r'\$[\d,.]+\s*(million|billion|m|b)',
        r'\d+\s*%',
        r'\d+x',
        r'doubled|tripled',
        r'\d+\s*(million|billion)\s*(dollar|user|customer)',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def extract_impact_quote(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        if has_quantified_impact(s) and has_ai_mention(s):
            return s.strip()[:200]
    for s in sentences:
        if has_ai_mention(s):
            return s.strip()[:200]
    return text[:200]


def main():
    rows = []

    for symbol, sector in COMPANIES:
        for quarter in QUARTERS:
            print(f"Fetching {symbol} {quarter}...")
            data = fetch_transcript(symbol, quarter)
            time.sleep(1.5)  # respect rate limits

            if not data or "transcript" not in data:
                print(f"  No transcript found for {symbol} {quarter}")
                rows.append({
                    "Company": symbol,
                    "Sector": sector,
                    "Quarter": quarter,
                    "AI Mentions": 0,
                    "Categories": "",
                    "Quantified Impact": "",
                    "Sample Quote": "No transcript available",
                    "Speaker": "",
                    "Title": "",
                })
                continue

            transcript = data["transcript"]
            ai_segments = [seg for seg in transcript if has_ai_mention(seg["content"])]
            print(f"  {len(transcript)} segments, {len(ai_segments)} mention AI")

            if not ai_segments:
                rows.append({
                    "Company": symbol,
                    "Sector": sector,
                    "Quarter": quarter,
                    "AI Mentions": 0,
                    "Categories": "",
                    "Quantified Impact": "No",
                    "Sample Quote": "No AI mentions found",
                    "Speaker": "",
                    "Title": "",
                })
                continue

            # Aggregate categories across all AI segments
            all_categories = set()
            any_quantified = False
            best_quote = ""
            best_speaker = ""
            best_title = ""

            for seg in ai_segments:
                cats = categorize(seg["content"])
                all_categories.update(cats)
                if has_quantified_impact(seg["content"]):
                    any_quantified = True
                    if not best_quote:
                        best_quote = extract_impact_quote(seg["content"])
                        best_speaker = seg.get("speaker", "")
                        best_title = seg.get("title", "")

            if not best_quote:
                seg = ai_segments[0]
                best_quote = extract_impact_quote(seg["content"])
                best_speaker = seg.get("speaker", "")
                best_title = seg.get("title", "")

            rows.append({
                "Company": symbol,
                "Sector": sector,
                "Quarter": quarter,
                "AI Mentions": len(ai_segments),
                "Categories": ", ".join(sorted(all_categories)),
                "Quantified Impact": "Yes" if any_quantified else "No",
                "Sample Quote": best_quote,
                "Speaker": best_speaker,
                "Title": best_title,
            })

    # Write CSV
    output_path = os.path.join(os.path.dirname(__file__), "ai_earnings_analysis.csv")
    fieldnames = ["Company", "Sector", "Quarter", "AI Mentions", "Categories", "Quantified Impact", "Sample Quote", "Speaker", "Title"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Wrote {len(rows)} rows to {output_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for row in rows:
        q = "Yes" if row["Quantified Impact"] == "Yes" else "No "
        print(f"{row['Company']:5s} {row['Quarter']}  AI mentions: {row['AI Mentions']:2d}  Quantified: {q}  Categories: {row['Categories']}")


if __name__ == "__main__":
    main()
