import csv
import json
import os
import re
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))

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


def has_real_quantified_impact(text):
    """Only count as quantified if numbers appear in the same sentence as AI keywords."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    patterns = [
        r'\$[\d,.]+\s*(million|billion|m\b|b\b)',
        r'\d+\s*%',
        r'\d+x\b',
        r'doubled|tripled|quadrupled',
        r'\d+\s*(million|billion)\s*(dollar|user|customer|parameter)',
    ]
    for s in sentences:
        if has_ai_mention(s):
            for p in patterns:
                if re.search(p, s, re.IGNORECASE):
                    return True
    return False


CATEGORY_RULES = {
    "AI-Native Product": [
        "our ai product", "ai platform", "ai service", "copilot", "ai offering",
        "ai-powered product", "ai solution", "our generative ai",
        "gpu", "accelerat", "inference", "training workload",
        "ai revenue", "ai arr", "ai buyer", "ai order", "ai demand",
        "ai backlog", "ai pipeline", "ai bookings", "shipped ai", "ai shipment",
        "ai pc", "ai server", "ai business", "ai portfolio",
        "ai guidance", "ai growth", "ai opportunity",
        "agent", "agentic", "ai model", "foundation model",
        "grok", "siri", "alexa", "cortana", "erica",
        "ai deal", "ai native", "ai tech", "ai front",
        "genai", "gen ai", "catalyst",
        "ai roadmap", "ai strategy",
    ],
    "AI-Enhanced Feature": [
        "added ai", "ai feature", "ai-powered", "ai-driven", "integrate ai",
        "leveraging ai", "using ai to", "ai capabilities in our",
        "recommendation", "personalization", "personalized ai", "ai search",
        "ai assistant", "ai-enabled", "ai connect", "illuminate",
        "plug-in", "plugin", "chatbot", "virtual assistant",
        "system of record", "smart", "intelligent",
        "predictive", "ai analytics", "ai insights",
        "digital twin", "digital platform", "digital ecosystem",
        "converting", "momentum driver", "embed",
        "apple intelligence", "now assist",
        "ai-first", "ai experience",
        "product experience", "photo select", "enhance", "user experience",
        "plumbed into", "securely adopting",
    ],
    "Internal Automation": [
        "internal", "automat", "productivity", "efficiency", "workflow",
        "coding assistant", "developer productivity", "customer service",
        "call center", "back office", "ai tool", "employee",
        "reduce cost", "cost saving", "cost reduction", "cost-conscious",
        "transcription", "localization", "code creation", "code generation",
        "preventive alert", "ai operations", "streamlin",
        "data science", "product analytics", "analytics capabilit",
        "sg&a", "process improvement", "technology and ai",
        "ai initiatives", "ai processes",
    ],
    "AI Infrastructure": [
        "data center", "cloud", "compute", "infrastructure", "hyperscal",
        "capital expenditure", "capex", "server", "chip", "semiconductor",
        "power", "energy demand", "cooling",
        "mi300", "mi250", "h100", "h200", "b100", "b200", "blackwell",
        "hardware", "networking", "interconnect", "ai spend", "ai investment",
        "ai workload", "ramp", "cluster",
    ],
    "Risk/Regulatory": [
        "regulat", "responsible ai", "ai governance", "ethical",
        "compliance", "ai risk", "ai safety", "bias", "privacy",
        "legislation", "oversight", "adversarial ai", "deepfake",
        "ai threat", "ai attack", "malicious",
    ],
}

VAGUE_KEYWORDS = [
    "excited about ai", "opportunity", "exploring", "potential of ai",
    "believe in ai", "ai is important", "ai journey",
    "follow up on ai", "comments on ai", "question on ai", "question is on ai",
    "asked about ai", "thoughts on ai",
    "ai transition", "leadership in", "focused on ai", "key area",
    "current time in ai", "era of ai", "world of ai", "age of ai",
    "love to ask about", "trying to think about", "wanted to get a sense",
    "questions on ai", "talk about ai", "topics",
    "convergence", "differentiator", "advancements in ai",
    "we do believe", "ai becomes",
    "impact on the demand", "impact the broader", "presents us with opportunit",
    "investments across", "perspective on", "narrative that ai",
    "conference", "presenting at",
]


def recategorize(text):
    """Improved categorization applied at build time."""
    lower = text.lower()
    matched = []
    for category, keywords in CATEGORY_RULES.items():
        if any(kw in lower for kw in keywords):
            matched.append(category)
    if not matched:
        if any(kw in lower for kw in VAGUE_KEYWORDS):
            return "Vague/Buzzword"
        return "Uncategorized"
    return ", ".join(matched)


def load_summary():
    rows = []
    path = os.path.join(DIR, "ai_summary.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["AI Mentions"] = int(row["AI Mentions"])
            row["Total Segments"] = int(row["Total Segments"])
            row["AI Intensity %"] = float(row["AI Intensity %"])
            rows.append(row)
    return rows


def load_quotes():
    """Load LLM-enriched quotes if available, fall back to CSV."""
    # Prefer v2 (with subcategories), then v1, then CSV
    for fname in ["ai_enriched_v2.json", "ai_enriched.json"]:
        enriched_path = os.path.join(DIR, fname)
        if os.path.exists(enriched_path):
            with open(enriched_path) as f:
                enriched = json.load(f)
            rows = []
            for v in enriched.values():
                rows.append({
                    "Company": v["company"],
                    "Sector": v["sector"],
                    "Quarter": v["quarter"],
                    "Speaker": v.get("speaker", ""),
                    "Role": v.get("role", ""),
                    "Categories": v.get("category", "Uncategorized"),
                    "Subcategory": v.get("subcategory", ""),
                    "Quote": v.get("original_quote", ""),
                    "Summary": v.get("summary", ""),
                    "Significance": v.get("significance", "Medium"),
                })
            print(f"  Using {fname}")
            return rows
    # Fallback to CSV
    rows = []
    path = os.path.join(DIR, "ai_quotes.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["Summary"] = ""
            row["Significance"] = "Medium"
            rows.append(row)
    return rows


SECTOR_MAP = {
    # Technology
    "Application Software": "Technology",
    "Systems Software": "Technology",
    "Technology Hardware, Storage & Peripherals": "Technology",
    "Semiconductors": "Technology",
    "Semiconductor Materials & Equipment": "Technology",
    "Electronic Equipment & Instruments": "Technology",
    "Electronic Manufacturing Services": "Technology",
    "Electronic Components": "Technology",
    "Communications Equipment": "Technology",
    "IT Consulting & Other Services": "Technology",
    "Data Processing & Outsourced Services": "Technology",
    "Internet Services & Infrastructure": "Technology",
    "Interactive Media & Services": "Technology",
    "Computer & Electronics Retail": "Technology",
    "Consumer Electronics": "Technology",
    "Electrical Components & Equipment": "Technology",
    # Banks
    "Diversified Banks": "Banks",
    "Regional Banks": "Banks",
    # Insurance
    "Life & Health Insurance": "Insurance",
    "Multi-line Insurance": "Insurance",
    "Property & Casualty Insurance": "Insurance",
    "Reinsurance": "Insurance",
    "Insurance Brokers": "Insurance",
    # Payments & Lending
    "Transaction & Payment Processing Services": "Payments & Lending",
    "Consumer Finance": "Payments & Lending",
    # Asset Management & Capital Markets
    "Asset Management & Custody Banks": "Asset Management & Capital Markets",
    "Investment Banking & Brokerage": "Asset Management & Capital Markets",
    "Financial Exchanges & Data": "Asset Management & Capital Markets",
    "Multi-Sector Holdings": "Asset Management & Capital Markets",
    # Healthcare
    "Biotechnology": "Healthcare",
    "Pharmaceuticals": "Healthcare",
    "Health Care Equipment": "Healthcare",
    "Health Care Facilities": "Healthcare",
    "Health Care Services": "Healthcare",
    "Health Care Supplies": "Healthcare",
    "Health Care Distributors": "Healthcare",
    "Health Care Technology": "Healthcare",
    "Health Care REITs": "Healthcare",
    "Life Sciences Tools & Services": "Healthcare",
    "Managed Health Care": "Healthcare",
    # Retail & Consumer
    "Broadline Retail": "Retail & Consumer",
    "Apparel Retail": "Retail & Consumer",
    "Automotive Retail": "Retail & Consumer",
    "Home Improvement Retail": "Retail & Consumer",
    "Homefurnishing Retail": "Retail & Consumer",
    "Consumer Staples Merchandise Retail": "Retail & Consumer",
    "Food Retail": "Retail & Consumer",
    "Food Distributors": "Retail & Consumer",
    "Hypermarkets & Super Centers": "Retail & Consumer",
    "Apparel, Accessories & Luxury Goods": "Retail & Consumer",
    "Footwear": "Retail & Consumer",
    "Household Products": "Retail & Consumer",
    "Personal Care Products": "Retail & Consumer",
    "Packaged Foods & Meats": "Retail & Consumer",
    "Soft Drinks & Non-alcoholic Beverages": "Retail & Consumer",
    "Tobacco": "Retail & Consumer",
    "Brewers": "Retail & Consumer",
    "Distillers & Vintners": "Retail & Consumer",
    "Restaurants": "Retail & Consumer",
    "Hotels, Resorts & Cruise Lines": "Retail & Consumer",
    "Casinos & Gaming": "Retail & Consumer",
    "Leisure Facilities": "Retail & Consumer",
    "Leisure Products": "Retail & Consumer",
    # Energy
    "Oil & Gas Exploration & Production": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Integrated Oil & Gas": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Storage & Transportation": "Energy",
    "Electric Utilities": "Energy & Utilities",
    "Gas Utilities": "Energy & Utilities",
    "Multi-Utilities": "Energy & Utilities",
    "Water Utilities": "Energy & Utilities",
    "Independent Power Producers & Energy Traders": "Energy & Utilities",
    "Renewable Electricity": "Energy & Utilities",
    # Industrials
    "Aerospace & Defense": "Industrials",
    "Industrial Conglomerates": "Industrials",
    "Construction Machinery & Heavy Transportation Equipment": "Industrials",
    "Construction & Engineering": "Industrials",
    "Construction Materials": "Industrials",
    "Building Products": "Industrials",
    "Agricultural & Farm Machinery": "Industrials",
    "Industrial Machinery & Supplies & Components": "Industrials",
    "Passenger Airlines": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Cargo Ground Transportation": "Industrials",
    "Railroads": "Industrials",
    "Passenger Ground Transportation": "Industrials",
    "Environmental & Facilities Services": "Industrials",
    "Diversified Support Services": "Industrials",
    "Research & Consulting Services": "Industrials",
    "Human Resource & Employment Services": "Industrials",
    "Trading Companies & Distributors": "Industrials",
    "Distributors": "Industrials",
    # Media & Telecom
    "Movies & Entertainment": "Media & Telecom",
    "Broadcasting": "Media & Telecom",
    "Cable & Satellite": "Media & Telecom",
    "Publishing": "Media & Telecom",
    "Advertising": "Media & Telecom",
    "Integrated Telecommunication Services": "Media & Telecom",
    "Alternative Carriers": "Media & Telecom",
    "Wireless Telecommunication Services": "Media & Telecom",
    # Real Estate
    "Data Center REITs": "Real Estate",
    "Industrial REITs": "Real Estate",
    "Retail REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Residential REITs": "Real Estate",
    "Specialized REITs": "Real Estate",
    "Telecom Tower REITs": "Real Estate",
    "Self-Storage REITs": "Real Estate",
    "Timber REITs": "Real Estate",
    "Real Estate Services": "Real Estate",
    # Materials
    "Commodity Chemicals": "Materials",
    "Diversified Chemicals": "Materials",
    "Specialty Chemicals": "Materials",
    "Fertilizers & Agricultural Chemicals": "Materials",
    "Agricultural Products & Services": "Materials",
    "Steel": "Materials",
    "Copper": "Materials",
    "Gold": "Materials",
    "Paper & Plastic Packaging Products & Materials": "Materials",
    "Industrial Gases": "Materials",
    "Metal, Glass & Plastic Containers": "Materials",
}


def map_sector(raw_sector):
    return SECTOR_MAP.get(raw_sector, raw_sector)


def esc(s):
    """Escape HTML entities."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html(summary, quotes):
    # Apply sector mapping
    for r in summary:
        r["Sector"] = map_sector(r["Sector"])
    for r in quotes:
        r["Sector"] = map_sector(r["Sector"])

    # Recompute Top Category for each summary row from recategorized quotes
    quote_cats = defaultdict(lambda: defaultdict(int))
    for q in quotes:
        key = (q["Company"], q["Quarter"])
        for cat in q["Categories"].split(", "):
            if cat and cat != "Uncategorized" and cat != "Vague/Buzzword":
                quote_cats[key][cat] += 1
    for r in summary:
        key = (r["Company"], r["Quarter"])
        if key in quote_cats and quote_cats[key]:
            r["Top Category"] = max(quote_cats[key], key=quote_cats[key].get)
        elif r["Best Quote"]:
            recat = recategorize(r["Best Quote"])
            r["Top Category"] = recat.split(", ")[0] if recat != "Uncategorized" else r["Top Category"]
    quarters = sorted(set(r["Quarter"] for r in summary))
    sectors = sorted(set(r["Sector"] for r in summary))
    companies = sorted(set(r["Company"] for r in summary))

    # 1. AI mentions over time (aggregate)
    mentions_by_q = defaultdict(int)
    intensity_by_q = defaultdict(list)
    for r in summary:
        mentions_by_q[r["Quarter"]] += r["AI Mentions"]
        if r["Total Segments"] > 0:
            intensity_by_q[r["Quarter"]].append(r["AI Intensity %"])

    timeline_labels = quarters
    timeline_mentions = [mentions_by_q[q] for q in quarters]
    timeline_intensity = [
        round(sum(intensity_by_q[q]) / len(intensity_by_q[q]), 1) if intensity_by_q[q] else 0
        for q in quarters
    ]

    # 2. Sector comparison (avg AI intensity, latest quarter)
    latest_q = quarters[-1] if quarters else ""
    sector_intensity = defaultdict(list)
    for r in summary:
        if r["Quarter"] == latest_q and r["Total Segments"] > 0:
            sector_intensity[r["Sector"]].append(r["AI Intensity %"])
    sector_data = sorted(
        [(s, round(sum(v) / len(v), 1)) for s, v in sector_intensity.items()],
        key=lambda x: x[1], reverse=True
    )[:20]

    # 3. Top 20 companies by AI intensity (latest quarter)
    company_latest = {}
    for r in summary:
        if r["Quarter"] == latest_q and r["AI Mentions"] > 0:
            company_latest[r["Company"]] = {
                "intensity": r["AI Intensity %"],
                "mentions": r["AI Mentions"],
                "sector": r["Sector"],
                "category": r["Top Category"],
            }
    top_companies = sorted(company_latest.items(), key=lambda x: x[1]["intensity"], reverse=True)[:20]

    # 4. Category distribution (all time)
    cat_counts = defaultdict(int)
    for r in quotes:
        for cat in r.get("Categories", "").split(", "):
            if cat:
                cat_counts[cat] += 1
    cat_total = sum(cat_counts.values())
    cat_data = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    cat_labels_with_pct = [
        f"{c[0]} ({round(c[1] / cat_total * 100, 1)}%)" if cat_total > 0 else c[0]
        for c in cat_data
    ]

    # 4b. Subcategory data for charts
    # Top 15 specific use cases (excluding Generic AI Mention)
    subcat_all = defaultdict(int)
    for r in quotes:
        sc = r.get("Subcategory", "")
        if sc and sc != "Generic AI Mention":
            subcat_all[sc] += 1
    top_subcats_chart = sorted(subcat_all.items(), key=lambda x: x[1], reverse=True)[:15]
    # All subcategories sorted by frequency (for filter dropdowns)
    all_subcats_sorted = sorted(subcat_all.items(), key=lambda x: x[1], reverse=True)

    # Subcategory by quarter timeline (top 8 subcategories over time)
    subcat_by_q = defaultdict(lambda: defaultdict(int))
    for r in quotes:
        sc = r.get("Subcategory", "")
        if sc and sc != "Generic AI Mention":
            subcat_by_q[r["Quarter"]][sc] += 1
    top8_subcats = [s[0] for s in top_subcats_chart[:8]]
    subcat_timeline = {}
    for sc in top8_subcats:
        subcat_timeline[sc] = [subcat_by_q[q].get(sc, 0) for q in quarters]

    # Nested doughnut: category -> subcategories mapping
    cat_subcat_map = defaultdict(lambda: defaultdict(int))
    for r in quotes:
        cat = r.get("Categories", "").split(", ")[0] if r.get("Categories") else ""
        sc = r.get("Subcategory", "")
        if cat and sc and sc != "Generic AI Mention":
            cat_subcat_map[cat][sc] += 1
    # Build nested doughnut data: inner ring = categories, outer ring = subcategories
    nested_inner = []  # [{label, value, color_idx}]
    nested_outer = []  # [{label, value, color_idx (parent)}]
    for idx, (cat, count) in enumerate(cat_data):
        cat_sc = cat_subcat_map.get(cat, {})
        # Inner ring: total non-generic subcategory mentions per category
        cat_sc_total = sum(cat_sc.values())
        if cat_sc_total > 0:
            nested_inner.append({"label": cat, "value": cat_sc_total, "color_idx": idx})
            # Outer ring: top subcategories for this category
            sorted_sc = sorted(cat_sc.items(), key=lambda x: x[1], reverse=True)[:5]
            for sc_name, sc_count in sorted_sc:
                nested_outer.append({"label": sc_name, "value": sc_count, "color_idx": idx})
            # "Other" bucket for remaining
            other_count = cat_sc_total - sum(s[1] for s in sorted_sc)
            if other_count > 0:
                nested_outer.append({"label": f"Other {cat}", "value": other_count, "color_idx": idx})

    # 5. Executive vs Analyst over time
    exec_by_q = defaultdict(int)
    analyst_by_q = defaultdict(int)
    for r in quotes:
        q = r["Quarter"]
        if r["Role"] == "Analyst":
            analyst_by_q[q] += 1
        else:
            exec_by_q[q] += 1
    exec_timeline = [exec_by_q[q] for q in quarters]
    analyst_timeline = [analyst_by_q[q] for q in quarters]

    # 6. Quantified impact — recalculate using stricter logic from quotes
    quant_by_q = defaultdict(lambda: [0, 0])  # [quantified, total]
    company_quarter_quant = defaultdict(bool)
    for r in quotes:
        key = f"{r['Company']}_{r['Quarter']}"
        if has_real_quantified_impact(r.get("Quote", "")):
            company_quarter_quant[key] = True

    for r in summary:
        if r["AI Mentions"] > 0:
            key = f"{r['Company']}_{r['Quarter']}"
            quant_by_q[r["Quarter"]][1] += 1
            if company_quarter_quant.get(key, False):
                quant_by_q[r["Quarter"]][0] += 1

    quant_rate = [
        round(quant_by_q[q][0] / quant_by_q[q][1] * 100, 1) if quant_by_q[q][1] > 0 else 0
        for q in quarters
    ]

    # 7. Top quotes — prioritize High significance from LLM, then quantified impact
    # Build lookup: (company, quarter) -> best enriched quote
    quote_lookup = {}
    for q in quotes:
        key = (q["Company"], q["Quarter"])
        sig = q.get("Significance", "Medium")
        sig_order = {"High": 3, "Medium": 2, "Low": 1}.get(sig, 0)
        existing = quote_lookup.get(key)
        if not existing or sig_order > existing["_sig_order"]:
            quote_lookup[key] = {**q, "_sig_order": sig_order}

    best_quotes = []
    for r in summary:
        if r["Best Quote"] and r["AI Mentions"] > 2:
            has_quant = has_real_quantified_impact(r["Best Quote"])
            lookup_key = (r["Company"], r["Quarter"])
            enriched_q = quote_lookup.get(lookup_key, {})
            best_quotes.append({
                "company": r["Company"],
                "sector": r["Sector"],
                "quarter": r["Quarter"],
                "intensity": r["AI Intensity %"],
                "quote": r["Best Quote"],
                "speaker": r["Speaker"],
                "quantified": has_quant,
                "summary": enriched_q.get("Summary", ""),
                "significance": enriched_q.get("Significance", "Medium"),
                "subcategory": enriched_q.get("Subcategory", ""),
                "category": enriched_q.get("Categories", ""),
            })
    best_quotes.sort(key=lambda x: (
        {"High": 3, "Medium": 2, "Low": 1}.get(x["significance"], 0),
        x["quantified"],
        x["intensity"]
    ), reverse=True)
    best_quotes = best_quotes[:30]  # more quotes since we have better data now

    # 8. Raw data for table tab — add best summary and subcategory from enriched quotes
    summary_lookup = {}
    subcat_lookup = {}
    for q in quotes:
        key = (q["Company"], q["Quarter"])
        sig = q.get("Significance", "Medium")
        sig_order = {"High": 3, "Medium": 2, "Low": 1}.get(sig, 0)
        existing = summary_lookup.get(key)
        if q.get("Summary") and (not existing or sig_order > existing[1]):
            summary_lookup[key] = (q["Summary"], sig_order)
        # Track most significant subcategory per company-quarter
        existing_sc = subcat_lookup.get(key)
        if q.get("Subcategory") and q["Subcategory"] != "Generic AI Mention" and (not existing_sc or sig_order > existing_sc[1]):
            subcat_lookup[key] = (q["Subcategory"], sig_order)

    raw_data_json = json.dumps([{
        "company": r["Company"],
        "sector": r["Sector"],
        "quarter": r["Quarter"],
        "mentions": r["AI Mentions"],
        "total": r["Total Segments"],
        "intensity": r["AI Intensity %"],
        "category": r["Top Category"],
        "subcategory": subcat_lookup.get((r["Company"], r["Quarter"]), ("", 0))[0],
        "who": r["Who Brought It Up"],
        "quantified": r["Quantified Impact"],
        "quote": r["Best Quote"],
        "speaker": r["Speaker"],
        "summary": summary_lookup.get((r["Company"], r["Quarter"]), ("", 0))[0],
    } for r in summary if r["AI Mentions"] > 0])

    # All summary data (including zero-mention rows) for company insights
    all_summary_json = json.dumps([{
        "company": r["Company"],
        "sector": r["Sector"],
        "quarter": r["Quarter"],
        "mentions": r["AI Mentions"],
        "total": r["Total Segments"],
        "intensity": r["AI Intensity %"],
        "category": r["Top Category"],
        "who": r["Who Brought It Up"],
        "quote": r["Best Quote"],
        "speaker": r["Speaker"],
    } for r in summary])

    # Quotes data for company-level insights
    quotes_json = json.dumps([{
        "company": r["Company"],
        "sector": r["Sector"],
        "quarter": r["Quarter"],
        "categories": r.get("Categories", ""),
        "subcategory": r.get("Subcategory", ""),
        "role": r.get("Role", ""),
        "quote": r.get("Quote", ""),
        "summary": r.get("Summary", ""),
        "significance": r.get("Significance", "Medium"),
    } for r in quotes])

    # 8b. Stock price data
    stock_prices_path = os.path.join(DIR, "stock_prices.json")
    stock_prices = {}
    if os.path.exists(stock_prices_path):
        with open(stock_prices_path) as f:
            stock_prices = json.load(f)

    # Build scatter data: AI intensity vs stock return for each company-quarter
    scatter_data = []
    momentum_data = []  # (ticker, quarter, ai_change, next_q_return)
    for r in summary:
        ticker = r["Company"]
        q = r["Quarter"]
        intensity = r["AI Intensity %"]
        if ticker in stock_prices and q in stock_prices[ticker]:
            ret = stock_prices[ticker][q]["return_pct"]
            scatter_data.append({
                "ticker": ticker,
                "sector": r["Sector"],
                "quarter": q,
                "intensity": intensity,
                "return_pct": ret,
                "mentions": r["AI Mentions"],
            })

    # AI momentum: did increasing AI intensity predict next quarter returns?
    sorted_quarters = sorted(quarters)
    q_to_next = {sorted_quarters[i]: sorted_quarters[i + 1] for i in range(len(sorted_quarters) - 1)}
    company_qdata = defaultdict(dict)
    for r in summary:
        company_qdata[r["Company"]][r["Quarter"]] = r
    for ticker, qdata in company_qdata.items():
        for q in sorted_quarters[:-1]:
            next_q = q_to_next[q]
            if q in qdata and next_q in qdata:
                curr_intensity = qdata[q]["AI Intensity %"]
                prev_intensity = qdata.get(sorted_quarters[max(0, sorted_quarters.index(q) - 1)], {}).get("AI Intensity %", 0)
                ai_change = curr_intensity - prev_intensity
                if ticker in stock_prices and next_q in stock_prices[ticker]:
                    next_ret = stock_prices[ticker][next_q]["return_pct"]
                    momentum_data.append({
                        "ticker": ticker,
                        "sector": company_qdata[ticker].get(q, {}).get("Sector", ""),
                        "quarter": q,
                        "ai_change": round(ai_change, 2),
                        "next_q_return": next_ret,
                    })

    # Aggregate momentum into buckets for bar chart
    increased_returns = [m["next_q_return"] for m in momentum_data if m["ai_change"] > 1]
    stable_returns = [m["next_q_return"] for m in momentum_data if -1 <= m["ai_change"] <= 1]
    decreased_returns = [m["next_q_return"] for m in momentum_data if m["ai_change"] < -1]

    momentum_buckets = json.dumps({
        "labels": ["Increased AI Talk\\n(>+1pp)", "Stable AI Talk\\n(±1pp)", "Decreased AI Talk\\n(<-1pp)"],
        "returns": [
            round(sum(increased_returns) / len(increased_returns), 2) if increased_returns else 0,
            round(sum(stable_returns) / len(stable_returns), 2) if stable_returns else 0,
            round(sum(decreased_returns) / len(decreased_returns), 2) if decreased_returns else 0,
        ],
        "counts": [len(increased_returns), len(stable_returns), len(decreased_returns)],
    })

    # Sector-level: avg return for high-AI vs low-AI companies per sector
    sector_ai_returns = defaultdict(lambda: {"high_ai": [], "low_ai": []})
    for d in scatter_data:
        bucket = "high_ai" if d["intensity"] > 5 else "low_ai"
        sector_ai_returns[d["sector"]][bucket].append(d["return_pct"])
    sector_comparison = []
    for sector in sorted(sector_ai_returns.keys()):
        hi = sector_ai_returns[sector]["high_ai"]
        lo = sector_ai_returns[sector]["low_ai"]
        if len(hi) >= 5 and len(lo) >= 5:
            sector_comparison.append({
                "sector": sector,
                "high_ai_return": round(sum(hi) / len(hi), 2),
                "low_ai_return": round(sum(lo) / len(lo), 2),
                "high_ai_count": len(hi),
                "low_ai_count": len(lo),
            })
    sector_comparison.sort(key=lambda x: x["high_ai_return"] - x["low_ai_return"], reverse=True)

    # Compute insight strings for the stock tab
    import math

    def pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return 0
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx2 = sum((x - mx) ** 2 for x in xs)
        dy2 = sum((y - my) ** 2 for y in ys)
        return num / math.sqrt(dx2 * dy2) if dx2 > 0 and dy2 > 0 else 0

    sc_xs = [d["intensity"] for d in scatter_data]
    sc_ys = [d["return_pct"] for d in scatter_data]
    sc_corr = pearson(sc_xs, sc_ys)
    hi_rets = [d["return_pct"] for d in scatter_data if d["intensity"] > 5]
    lo_rets = [d["return_pct"] for d in scatter_data if d["intensity"] <= 5]
    hi_avg = sum(hi_rets) / len(hi_rets) if hi_rets else 0
    lo_avg = sum(lo_rets) / len(lo_rets) if lo_rets else 0
    gap = hi_avg - lo_avg

    if abs(sc_corr) < 0.1:
        corr_verdict = f"Near-zero correlation ({sc_corr:.3f}) — AI talk alone doesn't predict stock returns."
    elif sc_corr > 0:
        corr_verdict = f"Weak positive correlation ({sc_corr:.3f}) — slight tendency for high-AI companies to outperform."
    else:
        corr_verdict = f"Weak negative correlation ({sc_corr:.3f}) — more AI talk slightly associated with lower returns."

    scatter_insight = (
        f"{corr_verdict} However, companies with AI Intensity >5% averaged "
        f"{hi_avg:.1f}% quarterly returns vs {lo_avg:.1f}% for those at ≤5% — "
        f"a {abs(gap):.1f}pp {'premium' if gap > 0 else 'discount'}. "
        f"This gap likely reflects that high-AI companies tend to be tech/growth stocks "
        f"that outperformed in 2023-2025, not necessarily that AI talk causes outperformance."
    )

    # Momentum insight
    inc_avg = sum(increased_returns) / len(increased_returns) if increased_returns else 0
    stab_avg = sum(stable_returns) / len(stable_returns) if stable_returns else 0
    dec_avg = sum(decreased_returns) / len(decreased_returns) if decreased_returns else 0
    best_bucket = max([(inc_avg, "increased"), (stab_avg, "stable"), (dec_avg, "decreased")], key=lambda x: x[0])

    momentum_insight = (
        f"Companies that increased AI talk averaged {inc_avg:.1f}% next-quarter returns, "
        f"stable companies averaged {stab_avg:.1f}%, and those that decreased averaged {dec_avg:.1f}%. "
    )
    if best_bucket[1] == "decreased":
        momentum_insight += (
            "Counterintuitively, companies that pulled back on AI talk saw the best subsequent returns. "
            "This may reflect mean reversion — after big AI-driven rallies, hype cools but fundamentals catch up."
        )
    elif best_bucket[1] == "increased":
        momentum_insight += (
            "Increasing AI focus correlated with better subsequent stock performance, "
            "suggesting the market rewards companies that are ramping up their AI efforts."
        )
    else:
        momentum_insight += (
            "Steady AI engagement correlated with the best returns — "
            "consistent AI strategy may signal operational maturity."
        )

    # Sector insight
    ai_premium_sectors = [s for s in sector_comparison if s["high_ai_return"] - s["low_ai_return"] > 3]
    ai_discount_sectors = [s for s in sector_comparison if s["high_ai_return"] - s["low_ai_return"] < -3]

    sector_insight = f"Of {len(sector_comparison)} sectors with enough data, "
    if ai_premium_sectors:
        top = sorted(ai_premium_sectors, key=lambda s: s["high_ai_return"] - s["low_ai_return"], reverse=True)[:3]
        names = ", ".join(s["sector"] for s in top)
        sector_insight += f"{len(ai_premium_sectors)} show an 'AI premium' (high-AI companies outperform). Strongest in: {names}. "
    if ai_discount_sectors:
        bottom = sorted(ai_discount_sectors, key=lambda s: s["high_ai_return"] - s["low_ai_return"])[:3]
        names = ", ".join(s["sector"] for s in bottom)
        sector_insight += f"{len(ai_discount_sectors)} show an 'AI discount' (high-AI companies underperform). Notably: {names}. "
    sector_insight += "The takeaway: AI's impact on stock performance is highly sector-dependent."

    stock_insights_json = json.dumps({
        "scatter": scatter_insight,
        "momentum": momentum_insight,
        "sector": sector_insight,
    })

    scatter_json = json.dumps(scatter_data)
    momentum_json = momentum_buckets
    sector_comp_json = json.dumps(sector_comparison)

    # Stats
    total_companies = len(companies)
    total_mentions = sum(r["AI Mentions"] for r in summary)
    total_with_ai = len([r for r in summary if r["AI Mentions"] > 0])
    total_transcripts = len(summary)

    # 9. Executive summary insights
    insights = []
    first_q = quarters[0] if quarters else ""

    # AI adoption rate
    companies_with_ai_first = len(set(r["Company"] for r in summary if r["Quarter"] == first_q and r["AI Mentions"] > 0))
    companies_total_first = len(set(r["Company"] for r in summary if r["Quarter"] == first_q))
    companies_with_ai_latest = len(set(r["Company"] for r in summary if r["Quarter"] == latest_q and r["AI Mentions"] > 0))
    companies_total_latest = len(set(r["Company"] for r in summary if r["Quarter"] == latest_q))
    if companies_total_first > 0 and companies_total_latest > 0:
        pct_first = round(companies_with_ai_first / companies_total_first * 100)
        pct_latest = round(companies_with_ai_latest / companies_total_latest * 100)
        insights.append({
            "title": "AI Adoption Across the S&P 500",
            "stat": f"{pct_latest}%",
            "detail": f"of companies mentioned AI in their {latest_q} earnings call, up from {pct_first}% in {first_q}.",
        })

    # Total mentions growth
    first_q_mentions = mentions_by_q.get(first_q, 0)
    latest_q_mentions = mentions_by_q.get(latest_q, 0)
    if first_q_mentions > 0:
        growth = round((latest_q_mentions - first_q_mentions) / first_q_mentions * 100)
        insights.append({
            "title": "AI Mentions Growth",
            "stat": f"{growth:+}%",
            "detail": f"Total AI mentions grew from {first_q_mentions:,} ({first_q}) to {latest_q_mentions:,} ({latest_q}) across all S&P 500 earnings calls.",
        })

    # Use case shift over time — compare category distribution in first vs latest quarter
    cat_by_q = defaultdict(lambda: defaultdict(int))
    for r in quotes:
        for cat in r.get("Categories", "").split(", "):
            if cat and cat != "Uncategorized":
                cat_by_q[r["Quarter"]][cat] += 1
    if first_q in cat_by_q and latest_q in cat_by_q:
        first_cats = cat_by_q[first_q]
        latest_cats = cat_by_q[latest_q]
        first_total_cat = sum(first_cats.values()) or 1
        latest_total_cat = sum(latest_cats.values()) or 1
        # Find fastest growing category
        cat_growth = []
        for cat in set(list(first_cats.keys()) + list(latest_cats.keys())):
            pct_f = first_cats.get(cat, 0) / first_total_cat * 100
            pct_l = latest_cats.get(cat, 0) / latest_total_cat * 100
            if pct_f > 3:  # only consider categories with meaningful presence
                cat_growth.append((cat, pct_f, pct_l, pct_l - pct_f))
        cat_growth.sort(key=lambda x: x[3], reverse=True)
        if cat_growth:
            rising = cat_growth[0]
            falling = cat_growth[-1]
            insights.append({
                "title": "How AI Use Cases Are Shifting",
                "stat": rising[0],
                "detail": f"is the fastest growing AI use case, rising from {rising[1]:.0f}% to {rising[2]:.0f}% of mentions ({first_q} to {latest_q}). Meanwhile, \"{falling[0]}\" fell from {falling[1]:.0f}% to {falling[2]:.0f}% — suggesting companies are moving past {falling[0].lower()} into {rising[0].lower()}.",
            })

    # Sector-specific use cases — what does each sector use AI for?
    sector_cat_counts = defaultdict(lambda: defaultdict(int))
    for r in quotes:
        for cat in r.get("Categories", "").split(", "):
            if cat and cat != "Uncategorized" and cat != "Vague/Buzzword":
                sector_cat_counts[r["Sector"]][cat] += 1
    sector_use_cases = []
    for sector in ["Technology", "Banks", "Insurance", "Payments & Lending", "Asset Management & Capital Markets", "Healthcare", "Retail & Consumer", "Industrials"]:
        if sector in sector_cat_counts:
            cats = sector_cat_counts[sector]
            if cats:
                top_use = max(cats, key=cats.get)
                total_s = sum(cats.values()) or 1
                top_pct = round(cats[top_use] / total_s * 100)
                sector_use_cases.append(f"{sector} = {top_use} ({top_pct}%)")
    if sector_use_cases:
        insights.append({
            "title": "AI Use Cases Vary by Sector",
            "stat": f"{len(sector_use_cases)} sectors",
            "detail": "Top AI use case by sector: " + ". ".join(sector_use_cases) + ".",
        })

    # Executive vs analyst — and how it's changing
    total_exec = sum(exec_timeline)
    total_analyst = sum(analyst_timeline)
    if total_exec + total_analyst > 0:
        exec_pct = round(total_exec / (total_exec + total_analyst) * 100)
        # Check if analysts are asking more over time
        first_q_analyst = analyst_by_q.get(first_q, 0)
        latest_q_analyst = analyst_by_q.get(latest_q, 0)
        analyst_trend = ""
        if first_q_analyst > 0:
            analyst_growth = round((latest_q_analyst - first_q_analyst) / first_q_analyst * 100)
            if analyst_growth > 20:
                analyst_trend = f" Analyst AI questions grew {analyst_growth}% — Wall Street is increasingly pressing companies on their AI strategy."
        insights.append({
            "title": "Who Drives the AI Conversation?",
            "stat": f"{exec_pct}% Executives",
            "detail": f"of AI mentions come from executives, not analysts. Companies are proactively talking about AI — it's not just analysts asking.{analyst_trend}",
        })

    # Quantified impact — what are companies actually claiming?
    quant_quotes_all = []
    for r in quotes:
        if has_real_quantified_impact(r.get("Quote", "")):
            quant_quotes_all.append(r)
    total_quant = len(quant_quotes_all)
    total_ai_quotes = len(quotes)
    if total_ai_quotes > 0:
        quant_pct = round(total_quant / total_ai_quotes * 100)
        insights.append({
            "title": "Quantified AI Impact Is Rare",
            "stat": f"{quant_pct}%",
            "detail": f"of AI mentions cite specific numbers. Only {total_quant:,} out of {total_ai_quotes:,} AI segments include measurable claims ($ revenue, % improvement, user counts). Most AI talk is still aspirational, not proven.",
        })

    # Biggest movers
    company_first_q_data = {}
    company_latest_q_data = {}
    for r in summary:
        if r["Quarter"] == first_q and r["AI Mentions"] > 0:
            company_first_q_data[r["Company"]] = {"intensity": r["AI Intensity %"], "sector": r["Sector"]}
        if r["Quarter"] == latest_q and r["AI Mentions"] > 0:
            company_latest_q_data[r["Company"]] = {"intensity": r["AI Intensity %"], "sector": r["Sector"]}
    movers = []
    for c in company_latest_q_data:
        if c in company_first_q_data:
            delta = company_latest_q_data[c]["intensity"] - company_first_q_data[c]["intensity"]
            movers.append((c, delta, company_first_q_data[c]["intensity"], company_latest_q_data[c]["intensity"], company_latest_q_data[c]["sector"]))
    movers.sort(key=lambda x: x[1], reverse=True)
    if len(movers) >= 3:
        top3 = movers[:3]
        detail_parts = [f"{m[0]} ({m[4]}, {m[2]:.0f}% to {m[3]:.0f}%)" for m in top3]
        insights.append({
            "title": "Fastest Growing AI Focus",
            "stat": top3[0][0],
            "detail": f"Top 3 companies by AI Intensity increase ({first_q} to {latest_q}): {', '.join(detail_parts)}. These companies dramatically increased how much of their earnings calls are about AI.",
        })

    # Companies that went from zero to AI
    new_adopters = []
    for c in company_latest_q_data:
        if c not in company_first_q_data:
            # Check if they had zero mentions in first quarter
            had_zero = any(r["Company"] == c and r["Quarter"] == first_q and r["AI Mentions"] == 0 for r in summary)
            if had_zero:
                new_adopters.append((c, company_latest_q_data[c]["intensity"], company_latest_q_data[c]["sector"]))
    new_adopters.sort(key=lambda x: x[1], reverse=True)
    if new_adopters:
        top_new = new_adopters[:5]
        names = ", ".join(f"{n[0]} ({n[2]})" for n in top_new)
        insights.append({
            "title": "New to the AI Conversation",
            "stat": f"{len(new_adopters)} companies",
            "detail": f"went from zero AI mentions in {first_q} to actively discussing AI in {latest_q}. Top newcomers: {names}.",
        })

    # Top company
    if top_companies:
        tc_name = top_companies[0][0]
        tc_intensity = top_companies[0][1]["intensity"]
        tc_sector = top_companies[0][1]["sector"]
        insights.append({
            "title": f"Highest AI Intensity ({latest_q})",
            "stat": f"{tc_name}",
            "detail": f"({tc_sector}) devoted {tc_intensity}% of their earnings call to AI — more than any other S&P 500 company in {latest_q}.",
            "subtitle": "AI Intensity = the % of an earnings call's segments that mention AI. 40% means nearly half the call discussed artificial intelligence.",
        })

    # Top specific AI use cases (from subcategories)
    subcat_counts = defaultdict(int)
    for q in quotes:
        sc = q.get("Subcategory", "")
        if sc and sc != "Generic AI Mention" and sc:
            subcat_counts[sc] += 1
    if subcat_counts:
        top_subcats = sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        subcat_detail = " | ".join(f"{sc[0]} ({sc[1]})" for sc in top_subcats)
        insights.append({
            "title": "What Companies Actually Do With AI",
            "stat": top_subcats[0][0],
            "detail": f"is the most common specific AI use case across the S&P 500. Top use cases: {subcat_detail}.",
        })

    # Pre-compute aggregate insights as JSON so JS can show them as default
    insights_json = json.dumps([{"title": i["title"], "stat": i["stat"], "detail": i["detail"], "subtitle": i.get("subtitle", "")} for i in insights] if insights else [])

    # Load ticker-to-name mapping
    ticker_names_path = os.path.join(DIR, "ticker_names.json")
    ticker_names = {}
    if os.path.exists(ticker_names_path):
        with open(ticker_names_path) as f:
            ticker_names = json.load(f)
    ticker_names_json = json.dumps(ticker_names)

    # Coverage: count transcripts per sector and per company
    # Total possible = companies_in_sector * len(quarters)
    transcripts_by_sector = defaultdict(int)
    companies_by_sector = defaultdict(set)
    for r in summary:
        transcripts_by_sector[r["Sector"]] += 1
        companies_by_sector[r["Sector"]].add(r["Company"])
    transcripts_by_company = defaultdict(int)
    for r in summary:
        transcripts_by_company[r["Company"]] += 1

    # Pre-compute rich sector data for research report view
    all_sectors = sorted(set(r["Sector"] for r in quotes if r["Sector"]))
    sector_report_data = {}
    for sector in all_sectors:
        sector_quotes = [r for r in quotes if r["Sector"] == sector]
        if not sector_quotes:
            continue
        sector_companies_set = set(r["Company"] for r in sector_quotes)
        sector_mentions = len(sector_quotes)
        sector_high = len([r for r in sector_quotes if r.get("Significance") == "High"])
        non_vague = [r for r in sector_quotes if r.get("Categories") != "Vague/Buzzword"]
        vague_ct = sector_mentions - len(non_vague)
        substance_pct = round(sector_high / len(non_vague) * 100) if non_vague else 0

        # Quarterly mention trends for narrative
        sq_by_q = defaultdict(int)
        for r in sector_quotes:
            sq_by_q[r["Quarter"]] += 1
        first_sq = sq_by_q.get(first_q, 0)
        latest_sq = sq_by_q.get(latest_q, 0)

        # Exec vs analyst
        exec_ct = len([r for r in sector_quotes if r.get("Role") != "Analyst"])
        analyst_ct = len([r for r in sector_quotes if r.get("Role") == "Analyst"])
        exec_pct = round(exec_ct / (exec_ct + analyst_ct) * 100) if (exec_ct + analyst_ct) > 0 else 0

        # Top companies by mention volume
        comp_mentions = defaultdict(int)
        for r in sector_quotes:
            comp_mentions[r["Company"]] += 1
        top_comps = sorted(comp_mentions.items(), key=lambda x: x[1], reverse=True)[:5]

        # Coverage
        sector_transcripts = transcripts_by_sector.get(sector, 0)
        sector_total_companies = len(companies_by_sector.get(sector, set()))
        sector_possible = sector_total_companies * len(quarters)

        # Build narrative as bullet points
        driver = "executives proactively bringing it up" if exec_pct > 65 else "analysts asking about it" if exec_pct < 35 else "both executives and analysts"
        bullets = [
            f"AI came up in {sector_mentions:,} earnings call segments across {len(sector_companies_set)} companies (each segment = one speaker's turn discussing AI).",
            f"Quarterly volume: {first_sq} segments in {first_q} &rarr; {latest_sq} in {latest_q}.",
            f"Who brings it up: {exec_pct}% from {driver}.",
            f"Of {sector_mentions:,} total segments, {vague_ct} were buzzword-only (e.g. 'AI is an opportunity'). Of the remaining {len(non_vague):,} substantive segments, {sector_high} ({substance_pct}%) cited concrete details — specific products, dollar figures, or measurable outcomes.",
        ]

        # Use cases: top 10 subcategories (excluding Generic/Vague)
        subcat_data = defaultdict(lambda: {"count": 0, "companies": set(), "quotes": []})
        for r in sector_quotes:
            sc = r.get("Subcategory", "")
            cat = r.get("Categories", "")
            if sc and sc != "Generic AI Mention" and cat != "Vague/Buzzword":
                subcat_data[sc]["count"] += 1
                subcat_data[sc]["companies"].add(r["Company"])
                subcat_data[sc]["quotes"].append(r)

        use_cases = []
        for sc_name, sc_info in sorted(subcat_data.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
            summaries = [r.get("Summary", "") for r in sc_info["quotes"]
                         if r.get("Significance") in ("Medium", "High") and r.get("Summary")]
            best_summary = max(summaries, key=len) if summaries else ""
            sorted_quotes = sorted(sc_info["quotes"],
                                   key=lambda r: (0 if r.get("Significance") == "High" else 1, -len(r.get("Summary", ""))))[:5]
            quote_list = []
            for q in sorted_quotes:
                quote_list.append({
                    "company": q["Company"],
                    "quarter": q["Quarter"],
                    "quote": q.get("Quote", "")[:400],
                    "summary": q.get("Summary", ""),
                    "role": q.get("Role", ""),
                    "significance": q.get("Significance", "Medium"),
                })
            concrete_total = len([r for r in sc_info["quotes"] if r.get("Significance") == "High"])
            use_cases.append({
                "name": sc_name,
                "count": sc_info["count"],
                "companyCount": len(sc_info["companies"]),
                "companies": sorted(sc_info["companies"]),
                "description": best_summary,
                "quotes": quote_list,
                "concreteCount": concrete_total,
            })

        sector_report_data[sector] = {
            "bullets": bullets,
            "coverage": f"Based on {sector_transcripts:,} earnings call transcripts from {sector_total_companies} companies ({sector_transcripts} of {sector_possible} possible company-quarters).",
            "stats": {
                "companies": len(sector_companies_set),
                "mentions": sector_mentions,
                "substancePct": substance_pct,
            },
            "topCompanies": [{"name": c[0], "mentions": c[1]} for c in top_comps],
            "useCases": use_cases,
        }

    sector_report_json = json.dumps(sector_report_data)

    # Aggregate narrative for "All" view
    total_high = len([r for r in quotes if r.get("Significance") == "High"])
    total_non_vague = len([r for r in quotes if r.get("Categories") != "Vague/Buzzword"])
    total_vague = len(quotes) - total_non_vague
    agg_substance = round(total_high / total_non_vague * 100) if total_non_vague else 0
    agg_exec = len([r for r in quotes if r.get("Role") != "Analyst"])
    agg_analyst = len([r for r in quotes if r.get("Role") == "Analyst"])
    agg_exec_pct = round(agg_exec / (agg_exec + agg_analyst) * 100) if (agg_exec + agg_analyst) > 0 else 0
    agg_first_mentions = mentions_by_q.get(first_q, 0)
    agg_latest_mentions = mentions_by_q.get(latest_q, 0)
    agg_possible = total_companies * len(quarters)
    agg_bullets = [
        f"AI came up in {total_mentions:,} earnings call segments across {total_companies} S&P 500 companies (each segment = one speaker's turn discussing AI).",
        f"Quarterly volume: {agg_first_mentions:,} segments in {first_q} &rarr; {agg_latest_mentions:,} in {latest_q}.",
        f"Who brings it up: {agg_exec_pct}% from executives (vs. analysts asking about it).",
        f"Of {total_mentions:,} total segments, {total_vague:,} were buzzword-only. Of the remaining {total_non_vague:,} substantive segments, {total_high:,} ({agg_substance}%) cited concrete details — specific products, dollar figures, or measurable outcomes.",
    ]
    agg_bullets_json = json.dumps(agg_bullets)
    agg_coverage = f"Based on {total_transcripts:,} earnings call transcripts from {total_companies} companies ({total_transcripts:,} of {agg_possible:,} possible company-quarters)."
    agg_coverage_json = json.dumps(agg_coverage)

    # Build dropdown options HTML
    sector_options_html = "".join(
        '<option value="{}">{}</option>'.format(esc(s), esc(s))
        for s in all_sectors
    )

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI in S&P 500 Earnings Calls (2023-2025)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root[data-theme="dark"] {{
    --bg: #0c0c0c;
    --bg-card: #161616;
    --bg-hover: #1e1e1e;
    --bg-input: #111111;
    --border: #262626;
    --text: #ebebeb;
    --text-secondary: #888888;
    --text-muted: #666666;
    --accent: #7c6aef;
    --accent2: #e85d75;
    --chart-grid: #262626;
    --chart-text: #888888;
  }}
  :root[data-theme="light"] {{
    --bg: #f8f8f6;
    --bg-card: #ffffff;
    --bg-hover: #f0f0ee;
    --bg-input: #ffffff;
    --border: #e4e4e0;
    --text: #1a1a1a;
    --text-secondary: #6b6b6b;
    --text-muted: #999999;
    --accent: #6c5ce7;
    --accent2: #e84393;
    --chart-grid: #e4e4e0;
    --chart-text: #6b6b6b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 40px 20px;
    transition: background 0.3s, color 0.3s;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 32px; }}
  .header-left {{ flex: 1; }}
  h1 {{
    font-size: 2.2rem;
    margin-bottom: 8px;
    color: var(--text);
  }}
  .subtitle {{ color: var(--text-muted); font-size: 1rem; }}
  .theme-toggle {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 8px 14px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
    white-space: nowrap;
    margin-top: 6px;
  }}
  .theme-toggle:hover {{ border-color: var(--accent); color: var(--text); }}
  .tabs {{
    display: flex;
    gap: 0;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
  }}
  .tab {{
    padding: 10px 20px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }}
  .tab:hover {{ color: var(--text); }}
  .tab.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .sector-selector {{
    margin-bottom: 20px;
  }}
  .sector-selector select {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 16px;
    border-radius: 10px;
    font-size: 0.95rem;
    cursor: pointer;
    min-width: 220px;
  }}
  .sector-selector select:focus {{
    outline: none;
    border-color: var(--accent);
  }}
  .insights-narrative {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .narrative-text {{
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.7;
    margin-bottom: 16px;
    list-style: none;
    padding: 0;
    margin-top: 0;
  }}
  .narrative-text li {{
    padding: 4px 0 4px 20px;
    position: relative;
  }}
  .narrative-text li::before {{
    content: '\\2022';
    position: absolute;
    left: 4px;
    color: var(--accent);
    font-weight: 700;
  }}
  .narrative-coverage {{
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }}
  .narrative-stats {{
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .narrative-stat {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }}
  .narrative-stat .num {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
  }}
  .narrative-stat .lbl {{
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .section-title {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
  }}
  .use-case-list {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .use-case-item {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }}
  .use-case-item:hover {{
    border-color: var(--accent);
    background: var(--bg-hover);
  }}
  .use-case-rank {{
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted);
    min-width: 24px;
    padding-top: 2px;
  }}
  .use-case-header {{
    flex: 1;
    min-width: 0;
  }}
  .use-case-name {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 4px;
  }}
  .use-case-count {{
    font-size: 0.8rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .use-case-desc {{
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.5;
  }}
  .use-case-companies {{
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
  }}
  .substance-badge {{
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 8px;
    vertical-align: middle;
    margin-left: 6px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
  }}
  .substance-badge.concrete {{
    background: rgba(72, 187, 120, 0.15);
    color: #48bb78;
    border: 1px solid rgba(72, 187, 120, 0.3);
  }}
  .substance-badge.none {{
    background: rgba(160, 160, 160, 0.1);
    color: var(--text-muted);
    border: 1px solid var(--border);
  }}
  .quote-drill-down {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }}
  .quote-item {{
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
  }}
  .quote-item:last-child {{
    border-bottom: none;
  }}
  .quote-item .quote-meta {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 6px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .quote-item .quote-summary {{
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 6px;
    font-weight: 500;
  }}
  .quote-item .quote-text {{
    font-size: 0.82rem;
    color: var(--text-muted);
    font-style: italic;
    line-height: 1.6;
  }}
  .quote-item .sig-high {{ color: var(--accent2); font-weight: 600; }}
  .quote-item .sig-medium {{ color: var(--text-secondary); }}
  .back-btn {{
    background: none;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.85rem;
    margin-bottom: 16px;
    transition: border-color 0.2s, color 0.2s;
  }}
  .back-btn:hover {{
    border-color: var(--accent);
    color: var(--accent);
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    transition: background 0.3s, border-color 0.3s;
  }}
  .stat-card .label {{ color: var(--text-muted); font-size: 0.85rem; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 32px;
  }}
  .chart-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    transition: background 0.3s, border-color 0.3s;
  }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-card h2 {{
    font-size: 1.05rem;
    margin-bottom: 4px;
    color: var(--text);
  }}
  .chart-subtitle {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 16px;
  }}
  .chart-insight {{
    margin-top: 16px;
    padding: 14px 18px;
    background: var(--bg-input);
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.55;
  }}
  .chart-insight strong {{ color: var(--text); }}
  .chart-wrap {{ position: relative; height: 300px; }}
  .chart-wrap.tall {{ height: 450px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }}
  th {{
    text-align: left;
    padding: 10px 12px;
    color: var(--text-muted);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  th:hover {{ color: var(--text); }}
  th .sort-arrow {{ margin-left: 4px; font-size: 0.7rem; }}
  td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  tr:hover {{ background: var(--bg-hover); }}
  .quote {{ color: var(--text-secondary); font-style: italic; max-width: 500px; }}
  .quote-speaker {{ color: var(--accent2); font-weight: 600; font-style: normal; }}
  .filters {{
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .filters select, .filters input {{
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    transition: border-color 0.2s;
  }}
  .filters select:focus, .filters input:focus {{
    outline: none;
    border-color: var(--accent);
  }}
  @media (max-width: 768px) {{
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-grid {{ grid-template-columns: 1fr; }}
    .filters {{ flex-direction: column; }}
    .header {{ flex-direction: column; gap: 12px; }}
  }}

  /* Onboarding */
  #onboarding {{
    position: fixed; inset: 0; z-index: 1000;
    background: var(--bg);
    display: flex; align-items: center; justify-content: center;
    transition: opacity 0.5s;
  }}
  #onboarding.hidden {{ opacity: 0; pointer-events: none; }}
  .onboard-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 48px 44px;
    max-width: 440px;
    width: 100%;
    text-align: center;
    box-shadow: 0 8px 40px rgba(0,0,0,0.08);
  }}
  .onboard-card h2 {{
    font-size: 1.8rem;
    margin-bottom: 6px;
  }}
  .onboard-card .onboard-sub {{
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 32px;
  }}
  .onboard-card input {{
    display: block;
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 1rem;
    margin-bottom: 16px;
    outline: none;
    transition: border-color 0.2s;
  }}
  .onboard-card input:focus {{ border-color: var(--accent); }}
  .onboard-card input::placeholder {{ color: var(--text-muted); }}
  .onboard-btn {{
    display: inline-block;
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 14px 40px;
    border-radius: 12px;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 8px;
    transition: transform 0.15s, box-shadow 0.2s;
    width: 100%;
  }}
  .onboard-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 20px rgba(108,92,231,0.35); }}
  .onboard-btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }}

  /* Loading screen */
  #loadingScreen {{
    position: fixed; inset: 0; z-index: 999;
    background: var(--bg);
    display: none; align-items: center; justify-content: center; flex-direction: column;
    transition: opacity 0.5s;
  }}
  #loadingScreen.active {{ display: flex; }}
  #loadingScreen.hidden {{ opacity: 0; pointer-events: none; }}
  .loading-text {{
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
  }}
  .loading-sub {{
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 32px;
  }}
  .loading-bar-track {{
    width: 280px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}
  .loading-bar {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), #2ec4b6, var(--accent2));
    background-size: 200% 100%;
    border-radius: 3px;
    animation: loadShimmer 1.5s ease-in-out infinite;
    transition: width 0.3s ease;
  }}
  @keyframes loadShimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
  }}
  .loading-steps {{
    margin-top: 28px;
    text-align: left;
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 2;
  }}
  .loading-steps .done {{ color: var(--accent); }}
  .loading-steps .active {{ color: var(--text); font-weight: 600; }}

  #mainDashboard {{ display: none; }}
  #mainDashboard.visible {{ display: block; }}
  .personalized {{ color: var(--accent); font-weight: 600; }}
</style>
</head>
<body>

<!-- ONBOARDING -->
<div id="onboarding">
  <div class="onboard-card">
    <h2>AI Earnings Tracker</h2>
    <p class="onboard-sub">See how S&P 500 companies are talking about AI across {len(quarters)} quarters of earnings calls.</p>
    <input type="text" id="onboardName" placeholder="Your first name" autocomplete="off" autofocus>
    <input type="text" id="onboardTicker" list="onboardTickerList" placeholder="A stock ticker you're curious about (e.g. AAPL)" autocomplete="off">
    <datalist id="onboardTickerList">
      {"".join(f'<option value="{esc(c)}">' for c in sorted(companies))}
    </datalist>
    <button class="onboard-btn" id="onboardGo" disabled onclick="startLoading()">Let's Go</button>
  </div>
</div>

<!-- LOADING SCREEN -->
<div id="loadingScreen">
  <div class="loading-text" id="loadingTitle">Crunching the numbers...</div>
  <div class="loading-sub" id="loadingSub"></div>
  <div class="loading-bar-track"><div class="loading-bar" id="loadingBar"></div></div>
  <div class="loading-steps" id="loadingSteps"></div>
</div>

<!-- MAIN DASHBOARD -->
<div id="mainDashboard">
<div class="container">
  <div class="header">
    <div class="header-left">
      <h1>AI in S&P 500 Earnings Calls</h1>
      <p class="subtitle" id="dashSubtitle">Analysis of {total_transcripts:,} earnings call transcripts across {total_companies} companies (2023-2025)</p>
    </div>
    <button class="theme-toggle" onclick="toggleTheme()">Dark Mode</button>
  </div>

  <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
    <input type="text" id="companySearch" list="companyList" placeholder="Search company ticker (e.g. AAPL)..." oninput="onCompanyFilter()" style="background:var(--bg-input); border:1px solid var(--border); color:var(--text); padding:10px 16px; border-radius:10px; font-size:0.95rem; width:320px;">
    <datalist id="companyList">
      {"".join(f'<option value="{esc(c)}">' for c in sorted(companies))}
    </datalist>
    <button onclick="document.getElementById('companySearch').value=''; onCompanyFilter();" class="theme-toggle" style="margin-top:0; font-size:0.8rem;">Show S&P 500</button>
    <span id="filterLabel" style="color:var(--text-muted); font-size:0.85rem;"></span>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('summary')">Key Insights</div>
    <div class="tab" onclick="switchTab('charts')">Charts</div>
    <div class="tab" onclick="switchTab('data')">Raw Data</div>
    <div class="tab" onclick="switchTab('quotes')">Top Quotes</div>
    <div class="tab" onclick="switchTab('stocks')">Stock vs AI</div>
  </div>

  <!-- KEY INSIGHTS TAB -->
  <div id="tab-summary" class="tab-content active">
    <div id="sectorSelector" class="sector-selector">
      <select id="sectorDropdown" onchange="selectSector(this.value)">
        <option value="All">All Sectors</option>
        {sector_options_html}
      </select>
    </div>
    <div id="narrativeSection" class="insights-narrative">
      <ul id="narrativeBullets" class="narrative-text"></ul>
      <div id="narrativeStats" class="narrative-stats"></div>
      <div id="narrativeCoverage" class="narrative-coverage"></div>
    </div>
    <div id="useCaseList" class="use-case-list"></div>
    <div id="quoteDrillDown" class="quote-drill-down" style="display:none;">
      <button class="back-btn" onclick="closeQuoteDrillDown()">&larr; Back to use cases</button>
      <h3 id="drillDownTitle" class="section-title"></h3>
      <div id="drillDownQuotes"></div>
    </div>
    <div id="companyInsightsStats" class="stats-grid" style="margin-bottom:24px; display:none;"></div>
    <div id="companyInsightsGrid" class="chart-grid" style="grid-template-columns: 1fr 1fr; display:none;"></div>
    <div id="companyTimeline" style="margin-top:24px; display:none;">
      <div class="chart-card full">
        <h2 id="companyTimelineTitle">AI Intensity Over Time</h2>
        <div class="chart-wrap"><canvas id="companyTimelineChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- CHARTS TAB -->
  <div id="tab-charts" class="tab-content">
    <div class="chart-grid">
      <div class="chart-card full">
        <h2>AI Mentions Across S&P 500 Over Time</h2>
        <p class="chart-subtitle">Total number of earnings call segments mentioning AI, and the average AI Intensity across all companies per quarter</p>
        <div class="chart-wrap"><canvas id="timelineChart"></canvas></div>
      </div>

      <div class="chart-card">
        <h2>Avg AI Intensity by Sector ({latest_q})</h2>
        <p class="chart-subtitle">AI Intensity = % of a company's earnings call segments that mention AI. Higher = more of the call is about AI.</p>
        <div class="chart-wrap tall"><canvas id="sectorChart"></canvas></div>
      </div>

      <div class="chart-card full">
        <h2>Top 15 Specific AI Use Cases</h2>
        <p class="chart-subtitle">Most common specific AI applications across S&P 500 earnings calls (excludes generic mentions)</p>
        <div class="chart-wrap tall"><canvas id="subcatBarChart"></canvas></div>
      </div>

      <div class="chart-card full">
        <h2>AI Use Case Evolution Over Time</h2>
        <p class="chart-subtitle">How the top AI use cases have grown or shifted quarter by quarter</p>
        <div class="chart-wrap tall"><canvas id="subcatTimelineChart"></canvas></div>
      </div>

      <div class="chart-card">
        <h2>Who Brings Up AI: Executives vs Analysts</h2>
        <p class="chart-subtitle">Are companies proactively talking about AI, or are analysts pushing the topic?</p>
        <div class="chart-wrap"><canvas id="roleChart"></canvas></div>
      </div>

      <div class="chart-card">
        <h2>Companies Citing Quantified AI Impact</h2>
        <p class="chart-subtitle">% of AI-mentioning companies that cite specific numbers (revenue, cost savings, users) in the same sentence as AI</p>
        <div class="chart-wrap"><canvas id="quantChart"></canvas></div>
      </div>

      <div class="chart-card full">
        <h2>Top 20 Companies by AI Intensity ({latest_q})</h2>
        <p class="chart-subtitle">Companies where the largest share of earnings call discussion is devoted to AI</p>
        <div class="chart-wrap tall"><canvas id="topCompaniesChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- RAW DATA TAB -->
  <div id="tab-data" class="tab-content">
    <div class="filters">
      <button onclick="downloadCSV()" class="theme-toggle" style="margin-top:0">Download CSV</button>
      <select id="filterSector" onchange="filterTable()">
        <option value="">All Sectors</option>
        {" ".join(f'<option value="{esc(s)}">{esc(s)}</option>' for s in sorted(sectors))}
      </select>
      <select id="filterQuarter" onchange="filterTable()">
        <option value="">All Quarters</option>
        {"".join(f'<option value="{q}">{q}</option>' for q in quarters)}
      </select>
      <select id="filterCategory" onchange="filterTable()">
        <option value="">All Categories</option>
        {"".join(f'<option value="{esc(c[0])}">{esc(c[0])}</option>' for c in cat_data)}
      </select>
      <select id="filterSubcategory" onchange="filterTable()">
        <option value="">All Subcategories</option>
        {"".join(f'<option value="{esc(s[0])}">{esc(s[0])} ({s[1]})</option>' for s in all_subcats_sorted)}
      </select>
    </div>
    <div style="overflow-x:auto">
    <table id="rawTable">
      <thead>
        <tr>
          <th onclick="sortTable(0)">Company <span class="sort-arrow"></span></th>
          <th onclick="sortTable(1)">Sector <span class="sort-arrow"></span></th>
          <th onclick="sortTable(2)">Quarter <span class="sort-arrow"></span></th>
          <th onclick="sortTable(3)">AI Mentions <span class="sort-arrow"></span></th>
          <th onclick="sortTable(4)">Intensity % <span class="sort-arrow"></span></th>
          <th onclick="sortTable(5)">Category <span class="sort-arrow"></span></th>
          <th onclick="sortTable(6)">What They're Doing <span class="sort-arrow"></span></th>
          <th onclick="sortTable(7)">Who <span class="sort-arrow"></span></th>
          <th>AI Summary</th>
          <th>Best Quote</th>
        </tr>
      </thead>
      <tbody id="rawTableBody">
      </tbody>
    </table>
    </div>
    <p id="rowCount" style="color:#7a776e; margin-top:12px; font-size:0.85rem;"></p>
  </div>

  <!-- QUOTES TAB -->
  <div id="tab-quotes" class="tab-content">
    <div class="filters" style="margin-bottom:12px;">
      <select id="filterQuotesSubcat" onchange="updateQuotes(document.getElementById('companySearch').value.toUpperCase().trim() || '')">
        <option value="">All Subcategories</option>
        {"".join(f'<option value="{esc(s[0])}">{esc(s[0])} ({s[1]})</option>' for s in all_subcats_sorted)}
      </select>
    </div>
    <div class="chart-card full">
      <h2 id="quotesTitle">Notable AI Quotes from Earnings Calls</h2>
      <p class="chart-subtitle" id="quotesSubtitle">Top AI quotes ranked by significance and impact. Each quote includes an AI-generated summary of how AI is being discussed.</p>
      <table>
        <thead>
          <tr>
            <th>Company</th>
            <th>Quarter</th>
            <th>Category</th>
            <th>What They're Doing</th>
            <th>Significance</th>
            <th>AI Summary</th>
            <th>Original Quote</th>
          </tr>
        </thead>
        <tbody id="quotesBody">
        </tbody>
      </table>
    </div>
  </div>

  <!-- STOCK VS AI TAB -->
  <div id="tab-stocks" class="tab-content">
    <div class="stats-grid" style="margin-bottom:24px">
      <div class="stat-card" id="stockCorr">
        <div class="stat-value">—</div>
        <div class="stat-label">Correlation</div>
      </div>
      <div class="stat-card" id="stockDataPts">
        <div class="stat-value">—</div>
        <div class="stat-label">Data Points</div>
      </div>
      <div class="stat-card" id="stockHighAI">
        <div class="stat-value">—</div>
        <div class="stat-label">High-AI Avg Return</div>
      </div>
      <div class="stat-card" id="stockLowAI">
        <div class="stat-value">—</div>
        <div class="stat-label">Low-AI Avg Return</div>
      </div>
    </div>
    <div class="chart-grid">
      <div class="chart-card full">
        <h2>AI Intensity vs Stock Return (Quarterly)</h2>
        <p class="chart-subtitle">Each dot is a company-quarter. X-axis: % of earnings call about AI. Y-axis: stock return that quarter. Does talking more about AI correlate with better stock performance?</p>
        <div class="chart-wrap tall"><canvas id="scatterChart"></canvas></div>
        <div class="chart-insight" id="scatterInsight"></div>
      </div>
      <div class="chart-card">
        <h2>AI Momentum Signal</h2>
        <p class="chart-subtitle">Companies that increased, maintained, or decreased their AI talk — what was their average stock return the <strong>following</strong> quarter?</p>
        <div class="chart-wrap tall"><canvas id="momentumChart"></canvas></div>
        <div class="chart-insight" id="momentumInsight"></div>
      </div>
      <div class="chart-card">
        <h2>High-AI vs Low-AI Returns by Sector</h2>
        <p class="chart-subtitle">Average quarterly return for companies with AI Intensity >5% vs ≤5%, broken down by sector. Does the "AI premium" vary by industry?</p>
        <div class="chart-wrap tall"><canvas id="sectorReturnChart"></canvas></div>
        <div class="chart-insight" id="sectorInsight"></div>
      </div>
    </div>
    <div id="stockCompanyDetail" style="margin-top:24px; display:none;">
      <div class="chart-card full">
        <h2 id="stockCompanyTitle">Stock Return vs AI Intensity</h2>
        <div class="chart-wrap"><canvas id="stockCompanyChart"></canvas></div>
        <div class="chart-insight" id="companyStockInsight"></div>
      </div>
    </div>
  </div>
</div>
</div> <!-- /mainDashboard -->

<script>
const RAW_DATA = {raw_data_json};
const ALL_SUMMARY = {all_summary_json};
const ALL_QUOTES = {quotes_json};
const AGG_INSIGHTS = {insights_json};
const SECTOR_DATA = {sector_report_json};
const AGG_BULLETS = {agg_bullets_json};
const AGG_COVERAGE = {agg_coverage_json};
const TICKER_NAMES = {ticker_names_json};
const TOTAL_MENTIONS = {total_mentions};
const TOTAL_WITH_AI = {total_with_ai};
const TOTAL_COMPANIES = {total_companies};
const QUARTERS = {json.dumps(quarters)};
const BEST_QUOTES = {json.dumps(best_quotes)};

// Aggregate chart data (pre-computed from Python)
const AGG_TIMELINE_LABELS = {json.dumps(timeline_labels)};
const AGG_TIMELINE_MENTIONS = {json.dumps(timeline_mentions)};
const AGG_TIMELINE_INTENSITY = {json.dumps(timeline_intensity)};
const AGG_SECTOR_DATA = {json.dumps(sector_data)};
const AGG_CAT_DATA = {json.dumps(cat_data)};
const AGG_CAT_LABELS = {json.dumps(cat_labels_with_pct)};
const AGG_EXEC_TIMELINE = {json.dumps(exec_timeline)};
const AGG_ANALYST_TIMELINE = {json.dumps(analyst_timeline)};
const AGG_QUANT_RATE = {json.dumps(quant_rate)};
const AGG_TOP_COMPANIES = {json.dumps([[c[0], c[1]["intensity"]] for c in top_companies])};

// Subcategory chart data
const TOP_SUBCATS = {json.dumps(top_subcats_chart)};
const SUBCAT_TIMELINE = {json.dumps(subcat_timeline)};
const SUBCAT_TIMELINE_LABELS = {json.dumps(top8_subcats)};
// Stock vs AI data
const SCATTER_DATA = {scatter_json};
const MOMENTUM_DATA = {momentum_json};
const SECTOR_COMP = {sector_comp_json};
const STOCK_INSIGHTS = {stock_insights_json};

function getThemeColors() {{
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {{
    grid: isDark ? '#262626' : '#e4e4e0',
    text: isDark ? '#888888' : '#6b6b6b',
  }};
}}

const tc = getThemeColors();
Chart.defaults.color = tc.text;
Chart.defaults.borderColor = tc.grid;

const violet = '#7c6aef';
const violetAlpha = 'rgba(124, 106, 239, 0.2)';
const pink = '#e85d75';
const pinkAlpha = 'rgba(232, 93, 117, 0.2)';
const teal = '#2ec4b6';
const tealAlpha = 'rgba(46, 196, 182, 0.2)';
const amber = '#f4a261';
const amberAlpha = 'rgba(244, 162, 97, 0.2)';
const electric = '#4dabf7';
const electricAlpha = 'rgba(77, 171, 247, 0.2)';
const lime = '#a3d977';
const limeAlpha = 'rgba(163, 217, 119, 0.2)';
const fuchsia = '#c084fc';

// Tabs
function switchTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector('[onclick="switchTab(\\'' + name + '\\')"]').classList.add('active');
  if (name === 'data') renderTable();
}}

// Chart instances
let chartTimeline, chartSector, chartRole, chartQuant, chartTopCompanies;
let chartSubcatBar, chartSubcatTimeline;

// Subcategory chart color palette — distinct hues for each category, lighter for outer ring
const CAT_COLORS = [violet, teal, amber, pink, electric, lime, fuchsia, '#ff9f43', '#54a0ff', '#5f27cd'];
const CAT_COLORS_LIGHT = [
  'rgba(124,106,239,0.55)', 'rgba(46,196,182,0.55)', 'rgba(244,162,97,0.55)',
  'rgba(232,93,117,0.55)', 'rgba(77,171,247,0.55)', 'rgba(163,217,119,0.55)',
  'rgba(192,132,252,0.55)', 'rgba(255,159,67,0.55)', 'rgba(84,160,255,0.55)', 'rgba(95,39,205,0.55)'
];
// 15 distinct colors for subcategory bar chart
const SUBCAT_BAR_COLORS = [
  '#7c6aef', '#2ec4b6', '#f4a261', '#e85d75', '#4dabf7',
  '#a3d977', '#c084fc', '#ff9f43', '#54a0ff', '#5f27cd',
  '#ee5a24', '#009432', '#0652DD', '#FDA7DF', '#C4E538'
];

function buildCharts(timelineLabels, timelineMentions, timelineIntensity, sectorLabels, sectorValues,
                     catLabels, catValues, execTimeline, analystTimeline, quantRate,
                     topLabels, topValues, titleSuffix) {{
  const tc = getThemeColors();

  if (chartTimeline) chartTimeline.destroy();
  if (chartSector) chartSector.destroy();
  if (chartRole) chartRole.destroy();
  if (chartQuant) chartQuant.destroy();
  if (chartTopCompanies) chartTopCompanies.destroy();

  chartTimeline = new Chart(document.getElementById('timelineChart'), {{
    type: 'bar',
    data: {{
      labels: timelineLabels,
      datasets: [
        {{ type: 'line', label: 'Avg AI Intensity %', data: timelineIntensity, borderColor: amber, backgroundColor: amberAlpha, yAxisID: 'y1', tension: 0.3, pointRadius: 5, pointBackgroundColor: amber }},
        {{ label: 'Total AI Mentions', data: timelineMentions, backgroundColor: violetAlpha, borderColor: violet, borderWidth: 1, borderRadius: 4, yAxisID: 'y' }},
      ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{
      y: {{ position: 'left', title: {{ display: true, text: 'Total Mentions' }} }},
      y1: {{ position: 'right', title: {{ display: true, text: 'Avg Intensity %' }}, grid: {{ drawOnChartArea: false }}, ticks: {{ callback: v => v + '%' }} }},
    }} }}
  }});

  // Sector chart — hide if filtering a single company
  const sectorCard = document.getElementById('sectorChart').closest('.chart-card');
  if (sectorLabels.length === 0) {{
    sectorCard.style.display = 'none';
    chartSector = new Chart(document.getElementById('sectorChart'), {{ type: 'bar', data: {{ labels: [], datasets: [] }} }});
  }} else {{
    sectorCard.style.display = '';
    chartSector = new Chart(document.getElementById('sectorChart'), {{
      type: 'bar',
      data: {{ labels: sectorLabels, datasets: [{{ label: 'Avg AI Intensity %', data: sectorValues, backgroundColor: tealAlpha, borderColor: teal, borderWidth: 1, borderRadius: 4 }}] }},
      options: {{ responsive: true, maintainAspectRatio: false, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ callback: v => v + '%' }} }} }} }}
    }});
  }}

  chartRole = new Chart(document.getElementById('roleChart'), {{
    type: 'bar',
    data: {{
      labels: timelineLabels,
      datasets: [
        {{ label: 'Executive', data: execTimeline, backgroundColor: 'rgba(124, 106, 239, 0.6)', borderColor: violet, borderWidth: 1, borderRadius: 4 }},
        {{ label: 'Analyst', data: analystTimeline, backgroundColor: 'rgba(46, 196, 182, 0.6)', borderColor: teal, borderWidth: 1, borderRadius: 4 }},
      ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }} }}
  }});

  chartQuant = new Chart(document.getElementById('quantChart'), {{
    type: 'line',
    data: {{ labels: timelineLabels, datasets: [{{ label: '% with Quantified Impact', data: quantRate, borderColor: pink, backgroundColor: pinkAlpha, fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: pink }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
  }});

  // Top companies chart — hide if filtering a single company
  const topCard = document.getElementById('topCompaniesChart').closest('.chart-card');
  if (topLabels.length === 0) {{
    topCard.style.display = 'none';
    chartTopCompanies = new Chart(document.getElementById('topCompaniesChart'), {{ type: 'bar', data: {{ labels: [], datasets: [] }} }});
  }} else {{
    topCard.style.display = '';
    chartTopCompanies = new Chart(document.getElementById('topCompaniesChart'), {{
      type: 'bar',
      data: {{ labels: topLabels, datasets: [{{ label: 'AI Intensity %', data: topValues, backgroundColor: violetAlpha, borderColor: violet, borderWidth: 1, borderRadius: 4 }}] }},
      options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ ticks: {{ callback: v => v + '%' }} }} }} }}
    }});
  }}

  // Update chart titles
  const suffix = titleSuffix ? ` — ${{titleSuffix}}` : '';
  document.querySelector('#tab-charts .chart-card.full h2').textContent = titleSuffix ? `${{titleSuffix}} AI Mentions Over Time` : 'AI Mentions Across S&P 500 Over Time';
}}

// --- Subcategory charts ---
function buildSubcatCharts(ticker) {{
  const tc = getThemeColors();
  if (chartSubcatBar) chartSubcatBar.destroy();
  if (chartSubcatTimeline) chartSubcatTimeline.destroy();

  const barCard = document.getElementById('subcatBarChart').closest('.chart-card');
  const timelineCard = document.getElementById('subcatTimelineChart').closest('.chart-card');

  if (!ticker) {{
    // --- AGGREGATE VIEW ---
    barCard.style.display = '';
    timelineCard.style.display = '';

    // Top 15 Use Cases Bar
    const barLabels = TOP_SUBCATS.map(s => s[0]);
    const barValues = TOP_SUBCATS.map(s => s[1]);
    chartSubcatBar = new Chart(document.getElementById('subcatBarChart'), {{
      type: 'bar',
      data: {{
        labels: barLabels,
        datasets: [{{
          label: 'Mentions',
          data: barValues,
          backgroundColor: barLabels.map((_, i) => SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length]),
          borderWidth: 0,
          borderRadius: 6,
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
          y: {{ grid: {{ display: false }}, ticks: {{ color: tc.text, font: {{ size: 11 }} }} }},
        }}
      }}
    }});

    // 3. Subcategory Evolution Timeline (stacked area)
    const tlDatasets = SUBCAT_TIMELINE_LABELS.map((sc, i) => ({{
      label: sc,
      data: SUBCAT_TIMELINE[sc],
      backgroundColor: SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length] + '44',
      borderColor: SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length],
      borderWidth: 2,
      fill: true,
      tension: 0.3,
      pointRadius: 3,
    }}));
    chartSubcatTimeline = new Chart(document.getElementById('subcatTimelineChart'), {{
      type: 'line',
      data: {{ labels: QUARTERS, datasets: tlDatasets }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'bottom', labels: {{ color: tc.text, padding: 10, font: {{ size: 10 }} }} }} }},
        scales: {{
          x: {{ grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
          y: {{ stacked: true, grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }}, title: {{ display: true, text: 'Mentions', color: tc.text }} }},
        }},
        interaction: {{ mode: 'index', intersect: false }},
      }}
    }});

  }} else {{
    // --- COMPANY VIEW ---
    const cQuotes = ALL_QUOTES.filter(r => r.company === ticker);

    // Company subcategory counts
    const cSubcats = {{}};
    cQuotes.forEach(q => {{
      const sc = q.subcategory;
      if (sc && sc !== 'Generic AI Mention') cSubcats[sc] = (cSubcats[sc] || 0) + 1;
    }});
    const cSubEntries = Object.entries(cSubcats).sort((a, b) => b[1] - a[1]);

    if (cSubEntries.length === 0) {{
      barCard.style.display = 'none';
      timelineCard.style.display = 'none';
      chartSubcatBar = new Chart(document.getElementById('subcatBarChart'), {{ type: 'bar', data: {{ labels: [], datasets: [] }} }});
      chartSubcatTimeline = new Chart(document.getElementById('subcatTimelineChart'), {{ type: 'line', data: {{ labels: [], datasets: [] }} }});
      return;
    }}

    // Company subcategory bar chart
    barCard.style.display = '';
    barCard.querySelector('h2').textContent = `${{ticker}} Specific AI Use Cases`;
    chartSubcatBar = new Chart(document.getElementById('subcatBarChart'), {{
      type: 'bar',
      data: {{
        labels: cSubEntries.map(s => s[0]),
        datasets: [{{
          label: 'Mentions',
          data: cSubEntries.map(s => s[1]),
          backgroundColor: cSubEntries.map((_, i) => SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length]),
          borderWidth: 0, borderRadius: 6,
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
          y: {{ grid: {{ display: false }}, ticks: {{ color: tc.text, font: {{ size: 11 }} }} }},
        }}
      }}
    }});

    // Company subcategory timeline
    const cScByQ = {{}};
    cQuotes.forEach(q => {{
      const sc = q.subcategory;
      if (sc && sc !== 'Generic AI Mention') {{
        if (!cScByQ[sc]) cScByQ[sc] = {{}};
        cScByQ[sc][q.quarter] = (cScByQ[sc][q.quarter] || 0) + 1;
      }}
    }});
    const cTopSc = cSubEntries.slice(0, 6).map(s => s[0]);
    const cTlDatasets = cTopSc.map((sc, i) => ({{
      label: sc,
      data: QUARTERS.map(q => cScByQ[sc]?.[q] || 0),
      backgroundColor: SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length] + '44',
      borderColor: SUBCAT_BAR_COLORS[i % SUBCAT_BAR_COLORS.length],
      borderWidth: 2, fill: true, tension: 0.3, pointRadius: 3,
    }}));

    if (cTopSc.length > 0) {{
      timelineCard.style.display = '';
      timelineCard.querySelector('h2').textContent = `${{ticker}} AI Use Cases Over Time`;
      chartSubcatTimeline = new Chart(document.getElementById('subcatTimelineChart'), {{
        type: 'line',
        data: {{ labels: QUARTERS, datasets: cTlDatasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ position: 'bottom', labels: {{ color: tc.text, padding: 10, font: {{ size: 10 }} }} }} }},
          scales: {{
            x: {{ grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
            y: {{ stacked: true, grid: {{ color: tc.grid }}, ticks: {{ color: tc.text, stepSize: 1 }} }},
          }},
          interaction: {{ mode: 'index', intersect: false }},
        }}
      }});
    }} else {{
      timelineCard.style.display = 'none';
      chartSubcatTimeline = new Chart(document.getElementById('subcatTimelineChart'), {{ type: 'line', data: {{ labels: [], datasets: [] }} }});
    }}
  }}
}}

// Initial chart render with aggregate data
buildCharts(AGG_TIMELINE_LABELS, AGG_TIMELINE_MENTIONS, AGG_TIMELINE_INTENSITY,
  AGG_SECTOR_DATA.map(s => s[0].substring(0, 30)), AGG_SECTOR_DATA.map(s => s[1]),
  AGG_CAT_LABELS, AGG_CAT_DATA.map(c => c[1]),
  AGG_EXEC_TIMELINE, AGG_ANALYST_TIMELINE, AGG_QUANT_RATE,
  AGG_TOP_COMPANIES.map(c => c[0]), AGG_TOP_COMPANIES.map(c => c[1]), '');
buildSubcatCharts('');

// Raw data table
let sortCol = 4;
let sortAsc = false;
let tableRendered = false;

function renderTable() {{
  if (tableRendered) return;
  tableRendered = true;
  filterTable();
}}

function filterTable() {{
  const companyFilter = document.getElementById('companySearch').value.toUpperCase().trim();
  const sectorFilter = document.getElementById('filterSector').value;
  const quarterFilter = document.getElementById('filterQuarter').value;
  const categoryFilter = document.getElementById('filterCategory').value;
  const subcatFilter = document.getElementById('filterSubcategory').value;

  let filtered = RAW_DATA.filter(r => {{
    if (companyFilter && !r.company.toUpperCase().includes(companyFilter)) return false;
    if (sectorFilter && r.sector !== sectorFilter) return false;
    if (quarterFilter && r.quarter !== quarterFilter) return false;
    if (categoryFilter && r.category !== categoryFilter) return false;
    if (subcatFilter && r.subcategory !== subcatFilter) return false;
    return true;
  }});

  // Sort
  filtered.sort((a, b) => {{
    let va, vb;
    switch(sortCol) {{
      case 0: va = a.company; vb = b.company; break;
      case 1: va = a.sector; vb = b.sector; break;
      case 2: va = a.quarter; vb = b.quarter; break;
      case 3: va = a.mentions; vb = b.mentions; break;
      case 4: va = a.intensity; vb = b.intensity; break;
      case 5: va = a.category; vb = b.category; break;
      case 6: va = a.subcategory || ''; vb = b.subcategory || ''; break;
      case 7: va = a.who; vb = b.who; break;
      default: va = a.company; vb = b.company;
    }}
    if (typeof va === 'string') {{
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }}
    return sortAsc ? va - vb : vb - va;
  }});

  const tbody = document.getElementById('rawTableBody');
  tbody.innerHTML = filtered.map(r => `
    <tr>
      <td style="font-weight:600">${{r.company}}</td>
      <td>${{r.sector}}</td>
      <td>${{r.quarter}}</td>
      <td>${{r.mentions}}</td>
      <td>${{r.intensity}}%</td>
      <td style="font-size:0.8rem;">${{r.category}}</td>
      <td style="font-size:0.8rem; font-weight:600; color:var(--accent);">${{r.subcategory || '-'}}</td>
      <td>${{r.who}}</td>
      <td style="max-width:300px;font-size:0.8rem;color:var(--text-secondary);">${{r.summary || '-'}}</td>
      <td class="quote" style="max-width:350px;font-size:0.8rem;">${{r.quote ? '"' + r.quote.substring(0, 200) + '..."' : '-'}}</td>
    </tr>
  `).join('');

  document.getElementById('rowCount').textContent = `Showing ${{filtered.length}} of ${{RAW_DATA.length}} entries`;
}}

function sortTable(col) {{
  if (sortCol === col) {{
    sortAsc = !sortAsc;
  }} else {{
    sortCol = col;
    sortAsc = col <= 2; // alphabetical ascending by default, numeric descending
  }}
  // Update sort arrows
  document.querySelectorAll('#rawTable th .sort-arrow').forEach((el, i) => {{
    el.textContent = i === col ? (sortAsc ? ' ▲' : ' ▼') : '';
  }});
  filterTable();
}}

// Company insights
let companyChart = null;

function makeCard(title, stat, detail, subtitle) {{
  return `<div class="chart-card">
    <p class="chart-subtitle" style="margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px; font-size:0.7rem;">${{title}}</p>
    <div style="font-size:2.2rem; font-weight:700; color:var(--accent); margin-bottom:8px;">${{stat}}</div>
    <p style="color:var(--text-secondary); line-height:1.6; font-size:0.9rem;">${{detail}}</p>
    ${{subtitle ? `<p style="color:var(--text-muted); font-size:0.78rem; margin-top:10px; line-height:1.5; border-top:1px solid var(--border); padding-top:10px;">${{subtitle}}</p>` : ''}}
  </div>`;
}}

const AI_INTENSITY_DEF = 'AI Intensity = the % of an earnings call\\'s segments that mention AI. 40% means nearly half the call discussed artificial intelligence.';

function esc(s) {{
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

function showUseCaseQuotes(name, quotes) {{
  const listEl = document.getElementById('useCaseList');
  const drillEl = document.getElementById('quoteDrillDown');
  const titleEl = document.getElementById('drillDownTitle');
  const quotesEl = document.getElementById('drillDownQuotes');
  listEl.style.display = 'none';
  drillEl.style.display = 'block';
  titleEl.textContent = name;
  quotesEl.innerHTML = quotes.map(q => `<div class="quote-item">
    <div class="quote-meta">
      <span><strong>${{esc(q.company)}}</strong></span>
      <span>${{esc(q.quarter)}}</span>
      <span>${{esc(q.role)}}</span>
      <span class="${{q.significance === 'High' ? 'sig-high' : 'sig-medium'}}">${{esc(q.significance)}}</span>
    </div>
    ${{q.summary ? `<div class="quote-summary">${{esc(q.summary)}}</div>` : ''}}
    <div class="quote-text">"${{esc(q.quote)}}"</div>
  </div>`).join('');
}}

function closeQuoteDrillDown() {{
  document.getElementById('useCaseList').style.display = 'flex';
  document.getElementById('quoteDrillDown').style.display = 'none';
}}

let _currentUseCases = [];

function renderUseCases(useCases) {{
  _currentUseCases = useCases || [];
  const el = document.getElementById('useCaseList');
  if (!useCases || useCases.length === 0) {{
    el.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No specific use cases found.</p>';
    return;
  }}
  el.innerHTML = '<h3 class="section-title" style="margin-bottom:8px;">Top AI Use Cases</h3>' +
    useCases.map((uc, i) => {{
      const comps = (uc.companies || []).slice(0, 8).map(t => TICKER_NAMES[t] ? `${{t}} (${{TICKER_NAMES[t]}})` : t).join(', ');
      const moreComps = (uc.companies || []).length > 8 ? ` + ${{uc.companies.length - 8}} more` : '';
      const concreteCount = uc.concreteCount != null ? uc.concreteCount : (uc.quotes || []).filter(q => q.significance === 'High').length;
      const concreteBadge = concreteCount > 0
        ? `<span class="substance-badge concrete" title="${{concreteCount}} quote${{concreteCount !== 1 ? 's' : ''}} with concrete details (specific products, dollar figures, measurable outcomes)">${{concreteCount}} concrete</span>`
        : `<span class="substance-badge none" title="No quotes with concrete details — mostly directional or aspirational">no concrete data</span>`;
      return `<div class="use-case-item" onclick="onUseCaseClick(${{i}})">
        <div class="use-case-rank">#${{i + 1}}</div>
        <div class="use-case-header">
          <div class="use-case-name">${{esc(uc.name)}} ${{concreteBadge}}</div>
          <div class="use-case-count">${{uc.companyCount > 1 ? uc.companyCount + ' companies · ' : ''}}${{uc.count}} segment${{uc.count !== 1 ? 's' : ''}}</div>
          ${{uc.description ? `<div class="use-case-desc">${{esc(uc.description)}}</div>` : ''}}
          ${{comps ? `<div class="use-case-companies">${{esc(comps)}}${{moreComps}}</div>` : ''}}
        </div>
      </div>`;
    }}).join('');
}}

function onUseCaseClick(idx) {{
  const uc = _currentUseCases[idx];
  if (uc) showUseCaseQuotes(uc.name, uc.quotes);
}}

function updateInsights() {{
  const ticker = document.getElementById('companySearch').value.toUpperCase().trim();
  const selectorEl = document.getElementById('sectorSelector');
  const narrativeEl = document.getElementById('narrativeSection');
  const useCaseEl = document.getElementById('useCaseList');
  const drillEl = document.getElementById('quoteDrillDown');
  const compStatsEl = document.getElementById('companyInsightsStats');
  const compGridEl = document.getElementById('companyInsightsGrid');
  const timelineEl = document.getElementById('companyTimeline');

  // Hide drill-down when switching views
  drillEl.style.display = 'none';

  if (!ticker || !ALL_SUMMARY.some(r => r.company === ticker)) {{
    // Show dropdown, narrative, use cases; hide company elements
    selectorEl.style.display = 'block';
    narrativeEl.style.display = 'block';
    useCaseEl.style.display = 'flex';
    compStatsEl.style.display = 'none';
    compGridEl.style.display = 'none';
    timelineEl.style.display = 'none';
    if (companyChart) {{ companyChart.destroy(); companyChart = null; }}

    if (activeSector && activeSector !== 'All' && SECTOR_DATA[activeSector]) {{
      // Sector view
      const sd = SECTOR_DATA[activeSector];
      document.getElementById('narrativeBullets').innerHTML = sd.bullets.map(b => `<li>${{b}}</li>`).join('');
      document.getElementById('narrativeStats').innerHTML = '';
      document.getElementById('narrativeCoverage').innerHTML = sd.coverage;
      renderUseCases(sd.useCases);
      return;
    }}

    // All view — aggregate narrative + top use cases computed client-side
    document.getElementById('narrativeBullets').innerHTML = AGG_BULLETS.map(b => `<li>${{b}}</li>`).join('');
    document.getElementById('narrativeStats').innerHTML = '';
    document.getElementById('narrativeCoverage').innerHTML = AGG_COVERAGE;

    // Compute top 12 use cases across all data
    const aggSubcat = {{}};
    ALL_QUOTES.forEach(q => {{
      const sc = q.subcategory;
      if (sc && sc !== 'Generic AI Mention' && q.categories !== 'Vague/Buzzword') {{
        if (!aggSubcat[sc]) aggSubcat[sc] = {{ count: 0, companies: new Set(), quotes: [] }};
        aggSubcat[sc].count++;
        aggSubcat[sc].companies.add(q.company);
        if (aggSubcat[sc].quotes.length < 5) {{
          aggSubcat[sc].quotes.push({{
            company: q.company, quarter: q.quarter, quote: (q.quote || '').substring(0, 400),
            summary: q.summary || '', role: q.role || '', significance: q.significance || 'Medium'
          }});
        }}
      }}
    }});
    const aggUseCases = Object.entries(aggSubcat)
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 12)
      .map(([name, d]) => ({{
        name,
        count: d.count,
        companyCount: d.companies.size,
        companies: [...d.companies].sort(),
        description: d.quotes.find(q => q.summary)?.summary || '',
        quotes: d.quotes.sort((a, b) => (a.significance === 'High' ? 0 : 1) - (b.significance === 'High' ? 0 : 1)),
      }}));
    renderUseCases(aggUseCases);
    return;
  }}

  // Company-specific view: use same narrative + use case layout
  selectorEl.style.display = 'none';
  narrativeEl.style.display = 'block';
  useCaseEl.style.display = 'flex';
  compStatsEl.style.display = 'none';
  compGridEl.style.display = 'none';

  const rows = ALL_SUMMARY.filter(r => r.company === ticker);
  const cQuotes = ALL_QUOTES.filter(r => r.company === ticker);
  const sector = rows[0]?.sector || '';
  const aiRows = rows.filter(r => r.mentions > 0);

  const totalMentions = rows.reduce((s, r) => s + r.mentions, 0);
  const maxIntensity = Math.max(...rows.map(r => r.intensity), 0);
  const latestRow = rows.filter(r => r.quarter === QUARTERS[QUARTERS.length - 1])[0];

  // Build company narrative as bullets
  const companyName = TICKER_NAMES[ticker] || ticker;
  const firstRow = rows.find(r => r.mentions > 0);
  const lastRow = [...aiRows].reverse()[0];

  let execCount = 0, analystCount = 0;
  cQuotes.forEach(q => {{ if (q.role === 'Analyst') analystCount++; else execCount++; }});
  const execPct = (execCount + analystCount) > 0 ? Math.round(execCount / (execCount + analystCount) * 100) : 0;
  const driverText = execPct > 65 ? 'executives proactively bringing it up' : execPct < 35 ? 'analysts asking about it' : 'both executives and analysts';

  const nonVague = cQuotes.filter(q => q.categories !== 'Vague/Buzzword');
  const highSig = cQuotes.filter(q => q.significance === 'High');
  const vagueCt = cQuotes.length - nonVague.length;
  const substanceRatio = nonVague.length >= 3 ? Math.round(highSig.length / nonVague.length * 100) : 0;

  const bullets = [];
  bullets.push(`${{esc(companyName)}} (${{esc(sector)}}) referenced AI in ${{totalMentions}} earnings call segments across ${{aiRows.length}} quarters.`);
  if (firstRow && lastRow && firstRow.quarter !== lastRow.quarter) {{
    bullets.push(`AI Intensity: ${{firstRow.intensity}}% (${{firstRow.quarter}}) &rarr; ${{lastRow.intensity}}% (${{lastRow.quarter}}). Peak: ${{maxIntensity}}%.`);
  }}
  // Sector comparison
  if (latestRow) {{
    const sectorPeers = ALL_SUMMARY.filter(r => r.sector === sector && r.quarter === latestRow.quarter);
    if (sectorPeers.length > 1) {{
      const sectorAvg = (sectorPeers.reduce((s, r) => s + r.intensity, 0) / sectorPeers.length).toFixed(1);
      const rank = sectorPeers.filter(r => r.intensity > latestRow.intensity).length + 1;
      bullets.push(`Ranked #${{rank}} of ${{sectorPeers.length}} in ${{esc(sector)}} by AI Intensity (${{latestRow.intensity}}% vs ${{sectorAvg}}% sector avg).`);
    }}
  }}
  bullets.push(`Who brings it up: ${{execPct}}% from ${{driverText}}.`);
  if (cQuotes.length > 0) {{
    bullets.push(`Of ${{cQuotes.length}} total segments, ${{vagueCt}} were buzzword-only. Of the remaining ${{nonVague.length}} substantive segments, ${{highSig.length}} (${{substanceRatio}}%) cited concrete details — specific products, dollar figures, or measurable outcomes.`);
  }}

  document.getElementById('narrativeBullets').innerHTML = bullets.map(b => `<li>${{b}}</li>`).join('');
  document.getElementById('narrativeStats').innerHTML = '';
  document.getElementById('narrativeCoverage').innerHTML = `Based on ${{rows.length}} earnings call transcripts (${{rows.length}} of ${{QUARTERS.length}} possible quarters).`;

  // Build use cases from company quotes
  const subcatData = {{}};
  cQuotes.forEach(q => {{
    const sc = q.subcategory;
    if (sc && sc !== 'Generic AI Mention' && q.categories !== 'Vague/Buzzword') {{
      if (!subcatData[sc]) subcatData[sc] = {{ count: 0, quotes: [] }};
      subcatData[sc].count++;
      subcatData[sc].quotes.push({{
        company: q.company, quarter: q.quarter,
        quote: (q.quote || '').substring(0, 400),
        summary: q.summary || '', role: q.role || '',
        significance: q.significance || 'Medium'
      }});
    }}
  }});
  const companyUseCases = Object.entries(subcatData)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 10)
    .map(([name, d]) => ({{
      name,
      count: d.count,
      companyCount: 1,
      companies: [ticker],
      description: d.quotes.find(q => q.summary)?.summary || '',
      quotes: d.quotes.sort((a, b) => (a.significance === 'High' ? 0 : 1) - (b.significance === 'High' ? 0 : 1)).slice(0, 5),
    }}));
  renderUseCases(companyUseCases);

  // Timeline chart
  timelineEl.style.display = 'block';
  document.getElementById('companyTimelineTitle').textContent = `${{ticker}} — AI Intensity Over Time`;
  const timelineData = QUARTERS.map(q => {{
    const row = rows.find(r => r.quarter === q);
    return row ? row.intensity : null;
  }});
  const mentionsData = QUARTERS.map(q => {{
    const row = rows.find(r => r.quarter === q);
    return row ? row.mentions : 0;
  }});

  if (companyChart) companyChart.destroy();
  const tc = getThemeColors();
  companyChart = new Chart(document.getElementById('companyTimelineChart'), {{
    type: 'bar',
    data: {{
      labels: QUARTERS,
      datasets: [
        {{
          type: 'line',
          label: 'AI Intensity %',
          data: timelineData,
          borderColor: violet,
          backgroundColor: violetAlpha,
          yAxisID: 'y1',
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: violet,
          spanGaps: true,
        }},
        {{
          label: 'AI Mentions',
          data: mentionsData,
          backgroundColor: amberAlpha,
          borderColor: amber,
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: 'y',
        }},
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        y: {{ position: 'left', title: {{ display: true, text: 'Mentions', color: tc.text }}, grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
        y1: {{ position: 'right', title: {{ display: true, text: 'Intensity %', color: tc.text }}, grid: {{ drawOnChartArea: false }}, ticks: {{ color: tc.text, callback: v => v + '%' }} }},
        x: {{ grid: {{ color: tc.grid }}, ticks: {{ color: tc.text }} }},
      }},
      plugins: {{ legend: {{ labels: {{ color: tc.text }} }} }},
    }}
  }});
}}

// --- Update Charts for a specific ticker ---
function updateCharts(ticker) {{
  if (!ticker) {{
    // Reset to S&P 500 aggregate
    buildCharts(AGG_TIMELINE_LABELS, AGG_TIMELINE_MENTIONS, AGG_TIMELINE_INTENSITY,
      AGG_SECTOR_DATA.map(s => s[0].substring(0, 30)), AGG_SECTOR_DATA.map(s => s[1]),
      AGG_CAT_LABELS, AGG_CAT_DATA.map(c => c[1]),
      AGG_EXEC_TIMELINE, AGG_ANALYST_TIMELINE, AGG_QUANT_RATE,
      AGG_TOP_COMPANIES.map(c => c[0]), AGG_TOP_COMPANIES.map(c => c[1]), '');
    buildSubcatCharts('');
    // Reset subcategory chart titles
    document.querySelector('#subcatBarChart').closest('.chart-card').querySelector('h2').textContent = 'Top 15 Specific AI Use Cases';
    document.querySelector('#subcatTimelineChart').closest('.chart-card').querySelector('h2').textContent = 'AI Use Case Evolution Over Time';
    return;
  }}

  const rows = ALL_SUMMARY.filter(r => r.company === ticker);
  const cQuotes = ALL_QUOTES.filter(r => r.company === ticker);
  if (rows.length === 0) return;

  // Timeline: mentions + intensity per quarter
  const tMentions = QUARTERS.map(q => {{ const r = rows.find(x => x.quarter === q); return r ? r.mentions : 0; }});
  const tIntensity = QUARTERS.map(q => {{ const r = rows.find(x => x.quarter === q); return r ? r.intensity : null; }});

  // Categories from quotes
  const catCounts = {{}};
  cQuotes.forEach(q => {{
    q.categories.split(', ').forEach(c => {{
      if (c && c !== 'Uncategorized') catCounts[c] = (catCounts[c] || 0) + 1;
    }});
  }});
  const catEntries = Object.entries(catCounts).sort((a, b) => b[1] - a[1]);
  const totalCats = catEntries.reduce((s, e) => s + e[1], 0) || 1;
  const catLabels = catEntries.map(c => `${{c[0]}} (${{Math.round(c[1]/totalCats*100)}}%)`);
  const catValues = catEntries.map(c => c[1]);

  // Exec vs Analyst per quarter
  const execByQ = QUARTERS.map(q => cQuotes.filter(r => r.quarter === q && r.role !== 'Analyst').length);
  const analystByQ = QUARTERS.map(q => cQuotes.filter(r => r.quarter === q && r.role === 'Analyst').length);

  // Quantified rate per quarter (using AI keyword + number heuristic)
  const aiKw = ['artificial intelligence',' ai ',' ai,',' ai.',' ai;','machine learning','generative ai','large language model',' llm','deep learning','neural network','copilot','chatgpt','natural language processing','computer vision'];
  function hasRealQuant(text) {{
    if (!text) return false;
    const sentences = text.split(/[.!?]+/);
    return sentences.some(s => {{
      const sl = (' ' + s.toLowerCase() + ' ');
      const hasAI = aiKw.some(k => sl.includes(k));
      const hasNum = /\\$[\\d,.]+|\\d+\\s*%|\\d+x\\b|doubled|tripled|\\d+\\s*(million|billion)/.test(s);
      return hasAI && hasNum;
    }});
  }}
  const quantByQ = QUARTERS.map(q => {{
    const qQuotes = cQuotes.filter(r => r.quarter === q);
    if (qQuotes.length === 0) return 0;
    const withQuant = qQuotes.filter(r => hasRealQuant(r.quote)).length;
    return Math.round(withQuant / qQuotes.length * 100);
  }});

  buildCharts(QUARTERS, tMentions, tIntensity,
    [], [], // no sector chart for single company
    catLabels, catValues,
    execByQ, analystByQ, quantByQ,
    [], [], // no top companies chart for single company
    ticker);
  buildSubcatCharts(ticker);
}}

// --- Update Quotes tab ---
function updateQuotes(ticker) {{
  const tbody = document.getElementById('quotesBody');
  const titleEl = document.getElementById('quotesTitle');
  const subtitleEl = document.getElementById('quotesSubtitle');

  const sigOrder = {{'High': 3, 'Medium': 2, 'Low': 1}};
  const sigColor = {{'High': 'var(--accent2)', 'Medium': 'var(--text-secondary)', 'Low': 'var(--text-muted)'}};

  const subcatFilter = document.getElementById('filterQuotesSubcat').value;

  let quotesToShow;
  if (!ticker) {{
    quotesToShow = BEST_QUOTES;
    titleEl.textContent = 'Notable AI Quotes from Earnings Calls';
    subtitleEl.textContent = 'Top AI quotes ranked by significance. Each includes an AI-generated summary of how AI is being discussed.';
  }} else {{
    // Get all enriched quotes for this company, sorted by significance then intensity
    const sigMap = {{'High': 3, 'Medium': 2, 'Low': 1}};
    quotesToShow = ALL_QUOTES.filter(r => r.company === ticker && r.quote)
      .sort((a, b) => (sigMap[b.significance] || 0) - (sigMap[a.significance] || 0) || 0)
      .map(r => ({{ company: r.company, quarter: r.quarter, sector: r.sector, quote: r.quote, summary: r.summary || '', significance: r.significance || 'Medium', speaker: '', category: r.categories || '', subcategory: r.subcategory || '' }}));
    titleEl.textContent = `${{ticker}} — AI Quotes from Earnings Calls`;
    subtitleEl.textContent = `All AI mentions for ${{ticker}} with AI-generated summaries, ranked by significance.`;
  }}

  // When no company is selected but subcategory filter is set, show all quotes matching that subcategory
  if (!ticker && subcatFilter) {{
    const sigMap = {{'High': 3, 'Medium': 2, 'Low': 1}};
    quotesToShow = ALL_QUOTES.filter(r => r.quote && r.subcategory === subcatFilter)
      .sort((a, b) => (sigMap[b.significance] || 0) - (sigMap[a.significance] || 0) || 0)
      .slice(0, 50)
      .map(r => ({{ company: r.company, quarter: r.quarter, sector: r.sector, quote: r.quote, summary: r.summary || '', significance: r.significance || 'Medium', speaker: '', category: r.categories || '', subcategory: r.subcategory || '' }}));
    titleEl.textContent = `AI Quotes — ${{subcatFilter}}`;
    subtitleEl.textContent = `Showing quotes categorized as "${{subcatFilter}}", ranked by significance.`;
  }} else if (subcatFilter) {{
    quotesToShow = quotesToShow.filter(q => q.subcategory === subcatFilter);
  }}

  const esc = s => s ? s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : '';
  tbody.innerHTML = quotesToShow.map(q => {{
    const sig = q.significance || 'Medium';
    const color = sigColor[sig] || 'var(--text-muted)';
    const catLabel = q.category || '';
    const subLabel = q.subcategory || '';
    return `<tr>
      <td style="font-weight:600">${{esc(q.company)}}</td>
      <td>${{esc(q.quarter)}}</td>
      <td style="font-size:0.8rem;">${{esc(catLabel)}}</td>
      <td style="font-size:0.8rem; font-weight:600; color:var(--accent);">${{esc(subLabel) || '<em style="color:var(--text-muted)">—</em>'}}</td>
      <td><span style="color:${{color}}; font-weight:600; font-size:0.8rem;">${{sig}}</span></td>
      <td style="font-size:0.85rem; color:var(--text-secondary); max-width:350px;">${{esc(q.summary) || '<em style="color:var(--text-muted)">—</em>'}}</td>
      <td class="quote" style="max-width:400px;">"${{esc(q.quote).substring(0, 250)}}${{q.quote && q.quote.length > 250 ? '...' : ''}}"${{q.speaker ? ' <span class="quote-speaker">— ' + esc(q.speaker) + '</span>' : ''}}</td>
    </tr>`;
  }}).join('');
}}

// --- Master filter handler ---
// --- STOCK VS AI ---
let chartScatter, chartMomentum, chartSectorReturn, chartStockCompany;

function pearsonCorrelation(data, xKey, yKey) {{
  const n = data.length;
  if (n < 3) return 0;
  const xs = data.map(d => d[xKey]), ys = data.map(d => d[yKey]);
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {{
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }}
  return dx2 > 0 && dy2 > 0 ? num / Math.sqrt(dx2 * dy2) : 0;
}}

function updateStocks(ticker) {{
  const tc = getThemeColors();
  if (chartScatter) chartScatter.destroy();
  if (chartMomentum) chartMomentum.destroy();
  if (chartSectorReturn) chartSectorReturn.destroy();
  if (chartStockCompany) chartStockCompany.destroy();

  const data = ticker ? SCATTER_DATA.filter(d => d.ticker === ticker) : SCATTER_DATA;
  const detailEl = document.getElementById('stockCompanyDetail');

  // Stats
  const corr = pearsonCorrelation(data, 'intensity', 'return_pct');
  const highAI = data.filter(d => d.intensity > 5);
  const lowAI = data.filter(d => d.intensity <= 5);
  const avgHigh = highAI.length > 0 ? (highAI.reduce((s, d) => s + d.return_pct, 0) / highAI.length).toFixed(1) : '—';
  const avgLow = lowAI.length > 0 ? (lowAI.reduce((s, d) => s + d.return_pct, 0) / lowAI.length).toFixed(1) : '—';

  document.querySelector('#stockCorr .stat-value').textContent = corr.toFixed(3);
  document.querySelector('#stockDataPts .stat-value').textContent = data.length.toLocaleString();
  document.querySelector('#stockHighAI .stat-value').textContent = avgHigh + '%';
  document.querySelector('#stockLowAI .stat-value').textContent = avgLow + '%';

  // Scatter plot
  const scatterPts = data.map(d => ({{ x: d.intensity, y: d.return_pct }}));
  chartScatter = new Chart(document.getElementById('scatterChart'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: ticker || 'S&P 500',
        data: scatterPts,
        backgroundColor: violetAlpha,
        borderColor: violet,
        pointRadius: ticker ? 5 : 2.5,
        pointHoverRadius: 6,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const d = data[ctx.dataIndex];
              return `${{d.ticker}} ${{d.quarter}}: AI ${{d.intensity}}%, Return ${{d.return_pct}}%`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'AI Intensity (%)', color: tc.textColor }},
          grid: {{ color: tc.gridColor }},
          ticks: {{ color: tc.textColor }},
        }},
        y: {{
          title: {{ display: true, text: 'Quarterly Stock Return (%)', color: tc.textColor }},
          grid: {{ color: tc.gridColor }},
          ticks: {{ color: tc.textColor }},
        }}
      }}
    }}
  }});

  // Momentum chart (aggregate only)
  document.getElementById('momentumChart').parentElement.parentElement.style.display = '';
  document.getElementById('sectorReturnChart').parentElement.parentElement.style.display = '';
  if (!ticker) {{
    chartMomentum = new Chart(document.getElementById('momentumChart'), {{
      type: 'bar',
      data: {{
        labels: MOMENTUM_DATA.labels,
        datasets: [{{
          label: 'Avg Next-Quarter Return (%)',
          data: MOMENTUM_DATA.returns,
          backgroundColor: [lime + 'cc', amberAlpha, fuchsia + 'cc'],
          borderColor: [lime, amber, fuchsia],
          borderWidth: 2,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterLabel: function(ctx) {{
                return `n = ${{MOMENTUM_DATA.counts[ctx.dataIndex]}} company-quarters`;
              }}
            }}
          }}
        }},
        scales: {{
          y: {{
            title: {{ display: true, text: 'Avg Next-Q Return (%)', color: tc.textColor }},
            grid: {{ color: tc.gridColor }},
            ticks: {{ color: tc.textColor }},
          }},
          x: {{
            grid: {{ display: false }},
            ticks: {{ color: tc.textColor }},
          }}
        }}
      }}
    }});

    // Sector comparison chart
    if (SECTOR_COMP.length > 0) {{
      const sLabels = SECTOR_COMP.map(s => s.sector);
      chartSectorReturn = new Chart(document.getElementById('sectorReturnChart'), {{
        type: 'bar',
        data: {{
          labels: sLabels,
          datasets: [
            {{
              label: 'High AI (>5%)',
              data: SECTOR_COMP.map(s => s.high_ai_return),
              backgroundColor: violetAlpha,
              borderColor: violet,
              borderWidth: 2,
            }},
            {{
              label: 'Low AI (≤5%)',
              data: SECTOR_COMP.map(s => s.low_ai_return),
              backgroundColor: amberAlpha,
              borderColor: amber,
              borderWidth: 2,
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: 'y',
          plugins: {{
            legend: {{ labels: {{ color: tc.textColor }} }},
            tooltip: {{
              callbacks: {{
                afterLabel: function(ctx) {{
                  const s = SECTOR_COMP[ctx.dataIndex];
                  return ctx.datasetIndex === 0 ? `n=${{s.high_ai_count}}` : `n=${{s.low_ai_count}}`;
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              title: {{ display: true, text: 'Avg Quarterly Return (%)', color: tc.textColor }},
              grid: {{ color: tc.gridColor }},
              ticks: {{ color: tc.textColor }},
            }},
            y: {{
              grid: {{ display: false }},
              ticks: {{ color: tc.textColor, font: {{ size: 11 }} }},
            }}
          }}
        }}
      }});
    }}
    // Aggregate insights
    document.getElementById('scatterInsight').innerHTML = `<strong>Finding:</strong> ${{STOCK_INSIGHTS.scatter}}`;
    document.getElementById('momentumInsight').innerHTML = `<strong>Finding:</strong> ${{STOCK_INSIGHTS.momentum}}`;
    document.getElementById('sectorInsight').innerHTML = `<strong>Finding:</strong> ${{STOCK_INSIGHTS.sector}}`;
    detailEl.style.display = 'none';
    document.getElementById('companyStockInsight').innerHTML = '';
  }} else {{
    // Company-specific: dual-axis line chart of AI intensity and stock return over time
    document.getElementById('momentumChart').parentElement.parentElement.style.display = 'none';
    document.getElementById('sectorReturnChart').parentElement.parentElement.style.display = 'none';
    detailEl.style.display = 'block';
    document.getElementById('stockCompanyTitle').textContent = `${{ticker}} — AI Intensity vs Stock Return Over Time`;

    const tData = SCATTER_DATA.filter(d => d.ticker === ticker).sort((a, b) => a.quarter.localeCompare(b.quarter));
    const labels = tData.map(d => d.quarter);
    const intensities = tData.map(d => d.intensity);
    const returns = tData.map(d => d.return_pct);

    chartStockCompany = new Chart(document.getElementById('stockCompanyChart'), {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [
          {{
            type: 'line',
            label: 'AI Intensity (%)',
            data: intensities,
            borderColor: violet,
            backgroundColor: violetAlpha,
            yAxisID: 'y1',
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: violet,
          }},
          {{
            label: 'Stock Return (%)',
            data: returns,
            backgroundColor: returns.map(r => r >= 0 ? limeAlpha : 'rgba(239, 68, 68, 0.3)'),
            borderColor: returns.map(r => r >= 0 ? lime : '#ef4444'),
            borderWidth: 2,
            yAxisID: 'y',
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: tc.textColor }} }},
        }},
        scales: {{
          y: {{
            position: 'left',
            title: {{ display: true, text: 'Stock Return (%)', color: tc.textColor }},
            grid: {{ color: tc.gridColor }},
            ticks: {{ color: tc.textColor }},
          }},
          y1: {{
            position: 'right',
            title: {{ display: true, text: 'AI Intensity (%)', color: tc.textColor }},
            grid: {{ display: false }},
            ticks: {{ color: tc.textColor }},
          }},
          x: {{
            grid: {{ display: false }},
            ticks: {{ color: tc.textColor }},
          }}
        }}
      }}
    }});

    // Company-specific scatter insight
    const cCorr = pearsonCorrelation(tData.map((d, i) => ({{ intensity: intensities[i], return_pct: returns[i] }})), 'intensity', 'return_pct');
    const totalReturn = returns.length > 0 ? returns.reduce((s, r) => s + r, 0).toFixed(1) : '0';
    const avgIntensity = intensities.length > 0 ? (intensities.reduce((s, r) => s + r, 0) / intensities.length).toFixed(1) : '0';
    const bestQ = tData.reduce((best, d) => d.return_pct > (best?.return_pct || -Infinity) ? d : best, null);
    const worstQ = tData.reduce((worst, d) => d.return_pct < (worst?.return_pct || Infinity) ? d : worst, null);

    let compInsight = `<strong>${{ticker}}</strong> has an average AI Intensity of ${{avgIntensity}}% with a cumulative return of ${{totalReturn}}% over ${{returns.length}} quarters. `;
    compInsight += `Correlation between AI intensity and stock return: ${{cCorr.toFixed(3)}}. `;
    if (bestQ) compInsight += `Best quarter: ${{bestQ.quarter}} (${{bestQ.return_pct > 0 ? '+' : ''}}${{bestQ.return_pct.toFixed(1)}}%). `;
    if (worstQ) compInsight += `Worst quarter: ${{worstQ.quarter}} (${{worstQ.return_pct > 0 ? '+' : ''}}${{worstQ.return_pct.toFixed(1)}}%). `;

    // Check if AI intensity trend matches stock trend
    const firstHalf = intensities.slice(0, Math.floor(intensities.length / 2));
    const secondHalf = intensities.slice(Math.floor(intensities.length / 2));
    const firstHalfRet = returns.slice(0, Math.floor(returns.length / 2));
    const secondHalfRet = returns.slice(Math.floor(returns.length / 2));
    const aiTrending = secondHalf.length > 0 && firstHalf.length > 0 &&
      (secondHalf.reduce((a,b) => a+b, 0) / secondHalf.length) > (firstHalf.reduce((a,b) => a+b, 0) / firstHalf.length);
    const retTrending = secondHalfRet.length > 0 && firstHalfRet.length > 0 &&
      (secondHalfRet.reduce((a,b) => a+b, 0) / secondHalfRet.length) > (firstHalfRet.reduce((a,b) => a+b, 0) / firstHalfRet.length);

    if (aiTrending && retTrending) {{
      compInsight += `Both AI intensity and stock returns have trended upward — AI focus and market performance are moving together.`;
    }} else if (aiTrending && !retTrending) {{
      compInsight += `AI intensity has increased but stock returns have softened — the market isn't yet rewarding ${{ticker}}'s growing AI focus.`;
    }} else if (!aiTrending && retTrending) {{
      compInsight += `Stock returns have improved despite steady/declining AI emphasis — ${{ticker}}'s performance is driven by other factors.`;
    }} else {{
      compInsight += `Both AI intensity and stock returns have been relatively flat or declining over this period.`;
    }}

    document.getElementById('companyStockInsight').innerHTML = compInsight;
    document.getElementById('scatterInsight').innerHTML = `<strong>Finding:</strong> ${{ticker}}'s correlation between AI talk and stock return is ${{cCorr.toFixed(3)}}. ${{Math.abs(cCorr) < 0.2 ? 'Weak relationship — other factors dominate.' : cCorr > 0 ? 'Positive link — quarters with more AI talk tend to have better returns.' : 'Negative link — more AI talk associated with weaker returns.'}}`;
    document.getElementById('momentumInsight').innerHTML = '';
    document.getElementById('sectorInsight').innerHTML = '';
  }}
}}

let activeSector = 'All';

function selectSector(sector) {{
  activeSector = sector;
  // Sync dropdown
  document.getElementById('sectorDropdown').value = sector;
  // Clear company search when switching sectors
  const searchEl = document.getElementById('companySearch');
  if (searchEl.value) {{
    searchEl.value = '';
    document.getElementById('filterLabel').textContent = '';
  }}
  updateInsights();
}}

function onCompanyFilter() {{
  const ticker = document.getElementById('companySearch').value.toUpperCase().trim();
  const valid = ticker && ALL_SUMMARY.some(r => r.company === ticker);
  const filterLabel = document.getElementById('filterLabel');

  if (valid) {{
    // Company search overrides sector filter — reset dropdown
    activeSector = 'All';
    document.getElementById('sectorDropdown').value = 'All';
    filterLabel.textContent = `Filtering: ${{ticker}}`;
    updateInsights();
    updateCharts(ticker);
    updateQuotes(ticker);
    updateStocks(ticker);
    if (tableRendered) filterTable();
  }} else if (!ticker) {{
    filterLabel.textContent = '';
    updateInsights();
    updateCharts('');
    updateQuotes('');
    updateStocks('');
    if (tableRendered) filterTable();
  }} else {{
    // Partial typing, don't update yet
    filterLabel.textContent = '';
  }}
}}

// Initialize on load
onCompanyFilter();

// Download CSV
function downloadCSV() {{
  const headers = ['Company','Sector','Quarter','AI Mentions','Total Segments','AI Intensity %','Top Category','Who Brought It Up','Best Quote','Speaker'];
  const escape = v => '"' + String(v || '').replace(/"/g, '""') + '"';
  const rows = RAW_DATA.map(r => [r.company, r.sector, r.quarter, r.mentions, r.total, r.intensity, r.category, r.who, r.quote, r.speaker].map(escape).join(','));
  const csv = [headers.join(','), ...rows].join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ai_sp500_earnings.csv';
  a.click();
  URL.revokeObjectURL(url);
}}

// Theme toggle
const allCharts = Chart.instances;
function toggleTheme() {{
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  document.querySelector('.theme-toggle').textContent = next === 'dark' ? 'Light Mode' : 'Dark Mode';
  const tc = getThemeColors();
  Object.values(Chart.instances).forEach(chart => {{
    chart.options.scales && Object.values(chart.options.scales).forEach(scale => {{
      if (scale.grid) scale.grid.color = tc.grid;
      if (scale.ticks) scale.ticks.color = tc.text;
      if (scale.title) scale.title.color = tc.text;
    }});
    if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {{
      chart.options.plugins.legend.labels.color = tc.text;
    }}
    chart.update();
  }});
  localStorage.setItem('theme', next);
}}
// Restore saved theme
(function() {{
  const saved = localStorage.getItem('theme');
  if (saved && saved !== document.documentElement.getAttribute('data-theme')) {{
    toggleTheme();
  }}
}})();

// --- Onboarding ---
const onboardName = document.getElementById('onboardName');
const onboardTicker = document.getElementById('onboardTicker');
const onboardGo = document.getElementById('onboardGo');

function checkOnboardReady() {{
  onboardGo.disabled = !onboardName.value.trim();
}}
onboardName.addEventListener('input', checkOnboardReady);
onboardName.addEventListener('keydown', e => {{ if (e.key === 'Enter' && !onboardGo.disabled) startLoading(); }});
onboardTicker.addEventListener('keydown', e => {{ if (e.key === 'Enter' && !onboardGo.disabled) startLoading(); }});

const LOADING_STEPS = [
  'Scanning {total_transcripts:,} earnings call transcripts...',
  'Detecting AI mentions across {total_companies} companies...',
  'Categorizing use cases and extracting quotes...',
  'Ranking companies by AI intensity...',
  'Comparing sectors and identifying trends...',
  'Generating your personalized insights...',
];

function startLoading() {{
  const name = onboardName.value.trim();
  const ticker = onboardTicker.value.toUpperCase().trim();

  // Hide onboarding
  document.getElementById('onboarding').classList.add('hidden');

  // Show loading
  const screen = document.getElementById('loadingScreen');
  const bar = document.getElementById('loadingBar');
  const stepsEl = document.getElementById('loadingSteps');
  const titleEl = document.getElementById('loadingTitle');
  const subEl = document.getElementById('loadingSub');

  titleEl.textContent = `Hold tight, ${{name}}...`;
  subEl.textContent = ticker ? `Preparing insights for ${{ticker}} and the S&P 500` : 'Analyzing the entire S&P 500';
  screen.classList.add('active');

  let step = 0;
  const totalSteps = LOADING_STEPS.length;
  const stepDuration = 420;

  function showStep() {{
    if (step >= totalSteps) {{
      bar.style.width = '100%';
      titleEl.textContent = `All set, ${{name}}!`;
      subEl.textContent = '';
      stepsEl.innerHTML = LOADING_STEPS.map(s => `<div class="done">${{s}}</div>`).join('');

      setTimeout(() => {{
        screen.classList.add('hidden');
        launchDashboard(name, ticker);
      }}, 600);
      return;
    }}

    const pct = Math.round(((step + 1) / totalSteps) * 100);
    bar.style.width = pct + '%';

    stepsEl.innerHTML = LOADING_STEPS.map((s, i) => {{
      if (i < step) return `<div class="done">${{s}}</div>`;
      if (i === step) return `<div class="active">${{s}}</div>`;
      return `<div>${{s}}</div>`;
    }}).join('');

    step++;
    setTimeout(showStep, stepDuration);
  }}
  showStep();
}}

function launchDashboard(name, ticker) {{
  // Save for session
  sessionStorage.setItem('onboarded', JSON.stringify({{ name, ticker }}));

  // Personalize
  const subtitle = document.getElementById('dashSubtitle');
  subtitle.innerHTML = `Specially made for <span class="personalized">${{name}}</span> &mdash; Analysis of {total_transcripts:,} transcripts across {total_companies} companies (2023-2025)`;

  // Show dashboard
  document.getElementById('mainDashboard').classList.add('visible');

  // Auto-filter to their ticker
  if (ticker && ALL_SUMMARY.some(r => r.company === ticker)) {{
    document.getElementById('companySearch').value = ticker;
    onCompanyFilter();
  }}
}}

// Skip onboarding if returning user
(function() {{
  const saved = sessionStorage.getItem('onboarded');
  if (saved) {{
    const data = JSON.parse(saved);
    document.getElementById('onboarding').style.display = 'none';
    document.getElementById('mainDashboard').classList.add('visible');
    if (data.name) {{
      const subtitle = document.getElementById('dashSubtitle');
      subtitle.innerHTML = `Specially made for <span class="personalized">${{data.name}}</span> &mdash; Analysis of {total_transcripts:,} transcripts across {total_companies} companies (2023-2025)`;
    }}
    if (data.ticker && ALL_SUMMARY.some(r => r.company === data.ticker)) {{
      document.getElementById('companySearch').value = data.ticker;
      onCompanyFilter();
    }}
    return;
  }}
}})();
</script>
</body>
</html>"""

    output_path = os.path.join(DIR, "dashboard.html")
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    summary = load_summary()
    quotes = load_quotes()
    print(f"Loaded {len(summary)} summary rows, {len(quotes)} quotes")
    build_html(summary, quotes)
