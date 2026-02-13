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
WORKERS = 15

SYSTEM_PROMPT = """You analyze earnings call quotes from S&P 500 companies. For each quote, return a JSON object with these fields:

1. "summary": 1-2 sentences describing EXACTLY what the quote says about AI. Paraphrase the quote directly — don't infer beyond what's stated. Start with what the company IS DOING, not what "the company discusses."

2. "category": The broad category. One of:
   - "AI Product": Company SELLS an AI product or service as revenue (AI SaaS, AI models, GPUs, AI platform)
   - "AI Feature": AI added INTO an existing non-AI product (AI search in their app, AI recommendations, AI assistant in their software)
   - "AI Adoption": Company USES AI tools internally to improve operations (AI coding tools, AI customer service, AI for back-office, productivity gains)
   - "AI Infrastructure": Building/investing in physical infrastructure FOR AI (data centers, chips, power, cloud compute, capex for AI)
   - "AI Strategy": Announcing AI plans, partnerships, investments, or acquisitions — but no shipped product yet
   - "AI Risk": AI-related risks, regulation, governance, safety, compliance
   - "Vague/Buzzword": Generic AI hype, no concrete product, plan, or use case mentioned

3. "subcategory": A specific 3-6 word label describing the EXACT use case. Examples:
   - "AI Code Generation" (using AI to write code)
   - "AI Voice Ordering" (voice AI for ordering food)
   - "AI Drug Discovery" (using AI to find drugs)
   - "AI Chip Sales" (selling AI chips/GPUs)
   - "AI Customer Service Bots" (AI chatbots for support)
   - "AI Content Generation" (generating marketing content)
   - "AI Search & Recommendations" (AI-powered search/recs)
   - "AI Data Center Expansion" (building data centers for AI)
   - "AI Fraud Detection" (using AI to detect fraud)
   - "AI Medical Imaging" (AI for radiology/diagnostics)
   - "AI Advertising Targeting" (AI for ad optimization)
   - "AI Supply Chain Optimization" (AI for logistics)
   - "AI Cybersecurity" (AI for threat detection)
   - "AI Autonomous Vehicles" (self-driving tech)
   - "AI Cloud Computing Platform" (selling AI cloud services)
   - "AI PC / AI Device" (AI-enabled hardware)
   - "AI Copilot / Assistant" (AI assistant in product)
   - "AI Agentic Workflows" (building AI agents)
   - "AI Revenue Growth" (reporting AI revenue numbers)
   - "AI Partnership / Acquisition" (strategic AI deal)
   - "AI Regulation / Governance" (AI policy/compliance)
   - "Generic AI Mention" (vague, no specifics)
   These are examples — create the most accurate label for each quote.

4. "significance": "High" (specific numbers, major launch, strategic shift), "Medium" (concrete but no metrics), "Low" (passing mention, vague)

Return ONLY valid JSON, no markdown."""


def call_openai(quote, retries=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Company: {quote["Company"]} ({quote["Sector"]}) | Quarter: {quote["Quarter"]} | Speaker: {quote.get("Role", "")}\n\nQuote: "{quote["Quote"]}"'},
    ]
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200,
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
                parsed = json.loads(content)
                # Validate required fields
                if "summary" in parsed and "category" in parsed and "subcategory" in parsed:
                    return parsed
                return parsed
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
            "subcategory": result.get("subcategory", ""),
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
            "category": "Uncategorized",
            "subcategory": "",
            "significance": "Medium",
        }


def main():
    quotes = []
    with open(os.path.join(DIR, "ai_quotes.csv"), newline="") as f:
        for row in csv.DictReader(f):
            quotes.append(row)
    print(f"Loaded {len(quotes)} quotes to enrich", flush=True)

    output_path = os.path.join(DIR, "ai_enriched_v2.json")
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

    if not work:
        print("Nothing to do!", flush=True)
        return

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

    # Print category/subcategory distribution
    cats = {}
    subcats = {}
    for v in enriched.values():
        c = v.get("category", "?")
        sc = v.get("subcategory", "?")
        cats[c] = cats.get(c, 0) + 1
        subcats[sc] = subcats.get(sc, 0) + 1
    print("\nCategories:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print(f"\nTop 20 sub-categories:")
    for sc, n in sorted(subcats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {sc}: {n}")


if __name__ == "__main__":
    main()
