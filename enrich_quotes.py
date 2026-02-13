import csv
import json
import os
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"
DIR = os.path.dirname(os.path.abspath(__file__))
WORKERS = 15  # parallel requests

SYSTEM_PROMPT = """You analyze earnings call quotes from S&P 500 companies. For each quote, return a JSON object with:

1. "summary": 1-2 sentence summary of HOW the company is using or discussing AI. Be specific about the product, use case, or impact.
2. "category": One of: "AI-Native Product", "AI-Enhanced Feature", "Internal Automation", "AI Infrastructure", "Risk/Regulatory", "Vague/Buzzword"
3. "significance": "High" (specific numbers, major launch, strategic shift), "Medium" (concrete but no numbers), "Low" (passing mention, vague)

Categories:
- AI-Native Product: Sells AI as core product (GPUs, AI platforms, AI models, AI SaaS, AI revenue)
- AI-Enhanced Feature: AI added to existing products (AI search, recommendations, AI assistant in their app)
- Internal Automation: Using AI internally (coding assistants, customer service bots, cost reduction, productivity)
- AI Infrastructure: Building infrastructure for AI (data centers, chips, power, cloud compute, capex)
- Risk/Regulatory: AI risks, regulation, governance, safety
- Vague/Buzzword: Generic hype with no specifics

Return ONLY valid JSON, no markdown."""


def call_openai(quote, retries=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Company: {quote["Company"]} ({quote["Sector"]}) | Quarter: {quote["Quarter"]} | Speaker: {quote.get("Role", "")}\n\n"{quote["Quote"]}"'},
    ]
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 150,
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 2, 30)
                time.sleep(wait)
            elif e.code >= 500:
                time.sleep(2 ** attempt)
            else:
                try:
                    e.read()
                except:
                    pass
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def process_one(args):
    i, key, q = args
    result = call_openai(q)
    if result:
        return key, {
            "company": q["Company"],
            "sector": q["Sector"],
            "quarter": q["Quarter"],
            "speaker": q.get("Speaker", ""),
            "role": q.get("Role", ""),
            "original_quote": q["Quote"],
            "summary": result.get("summary", ""),
            "category": result.get("category", "Uncategorized"),
            "significance": result.get("significance", "Medium"),
        }
    else:
        return key, {
            "company": q["Company"],
            "sector": q["Sector"],
            "quarter": q["Quarter"],
            "speaker": q.get("Speaker", ""),
            "role": q.get("Role", ""),
            "original_quote": q["Quote"],
            "summary": "",
            "category": q.get("Categories", "Uncategorized"),
            "significance": "Medium",
        }


def main():
    quotes = []
    with open(os.path.join(DIR, "ai_quotes.csv"), newline="") as f:
        for row in csv.DictReader(f):
            quotes.append(row)
    print(f"Loaded {len(quotes)} quotes to enrich", flush=True)

    output_path = os.path.join(DIR, "ai_enriched.json")
    enriched = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            enriched = json.load(f)
        print(f"Resuming: {len(enriched)} already done", flush=True)

    # Build work queue
    work = []
    for i, q in enumerate(quotes):
        key = f"{q['Company']}_{q['Quarter']}_{i}"
        if key not in enriched:
            work.append((i, key, q))
    print(f"Remaining: {len(work)} to process with {WORKERS} workers", flush=True)

    processed = 0
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_one, item): item for item in work}

        for future in as_completed(futures):
            key, result = future.result()
            enriched[key] = result
            if result["summary"]:
                processed += 1
            else:
                errors += 1

            total_done = processed + errors
            if total_done % 100 == 0:
                with open(output_path, "w") as f:
                    json.dump(enriched, f)
                elapsed = time.time() - start
                rate = total_done / elapsed if elapsed > 0 else 0
                remaining = (len(work) - total_done) / rate / 60 if rate > 0 else 0
                print(f"[{len(enriched)}/{len(quotes)}] {processed} ok, {errors} err, {rate:.1f}/sec, ~{remaining:.0f} min left", flush=True)

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone! {processed} enriched, {errors} errors in {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
