import json
import time
import urllib.request
import os

API_KEY = "JNGL6W6EOSQXDK01"
BASE_URL = "https://www.alphavantage.co/query"
DIR = os.path.dirname(os.path.abspath(__file__))

# Quarter -> end-of-quarter month
QUARTER_MONTHS = {
    "2023Q1": "2023-03", "2023Q2": "2023-06", "2023Q3": "2023-09", "2023Q4": "2023-12",
    "2024Q1": "2024-03", "2024Q2": "2024-06", "2024Q3": "2024-09", "2024Q4": "2024-12",
    "2025Q1": "2025-03", "2025Q2": "2025-06", "2025Q3": "2025-09", "2025Q4": "2025-12",
}
# Previous quarter for return calculation
PREV_QUARTER_MONTH = {
    "2023Q1": "2022-12", "2023Q2": "2023-03", "2023Q3": "2023-06", "2023Q4": "2023-09",
    "2024Q1": "2023-12", "2024Q2": "2024-03", "2024Q3": "2024-06", "2024Q4": "2024-09",
    "2025Q1": "2024-12", "2025Q2": "2025-03", "2025Q3": "2025-06", "2025Q4": "2025-09",
}


def fetch_monthly(symbol):
    url = f"{BASE_URL}?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "grow-1k-research"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("Monthly Adjusted Time Series", {})
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return {}


def find_month_close(monthly_data, target_month):
    """Find the adjusted close price for a target month (YYYY-MM).
    Uses split/dividend-adjusted close to avoid distortions."""
    for date_str, values in monthly_data.items():
        if date_str.startswith(target_month):
            return float(values["5. adjusted close"])
    return None


def main():
    with open(os.path.join(DIR, "sp500.json")) as f:
        companies = json.load(f)

    output_path = os.path.join(DIR, "stock_prices.json")

    # Resume support
    prices = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            prices = json.load(f)
        print(f"Resuming: {len(prices)} tickers already done")

    remaining = [c for c in companies if c["symbol"] not in prices]
    print(f"Pulling monthly prices for {len(remaining)} tickers ({len(companies)} total)")
    print(f"Rate: ~70/min, estimated: ~{len(remaining) // 70 + 1} minutes\n")

    calls = 0
    calls_this_minute = 0
    minute_start = time.time()

    for i, company in enumerate(remaining):
        symbol = company["symbol"]

        # Rate limiting
        calls_this_minute += 1
        if calls_this_minute >= 70:
            elapsed = time.time() - minute_start
            if elapsed < 62:
                wait = 62 - elapsed
                print(f"  Rate limit pause: {wait:.0f}s")
                time.sleep(wait)
            calls_this_minute = 0
            minute_start = time.time()

        monthly = fetch_monthly(symbol)
        calls += 1

        if not monthly:
            prices[symbol] = {}
        else:
            quarterly_returns = {}
            for quarter, end_month in QUARTER_MONTHS.items():
                prev_month = PREV_QUARTER_MONTH[quarter]
                end_close = find_month_close(monthly, end_month)
                prev_close = find_month_close(monthly, prev_month)
                if end_close and prev_close and prev_close > 0:
                    ret = round((end_close - prev_close) / prev_close * 100, 2)
                    quarterly_returns[quarter] = {
                        "return_pct": ret,
                        "close": end_close,
                        "prev_close": prev_close,
                    }
            prices[symbol] = quarterly_returns

        # Progress + save every 50
        if calls % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(prices, f)
            pct = (len(prices)) / len(companies) * 100
            print(f"[{pct:.0f}%] {len(prices)}/{len(companies)} tickers, {calls} calls")

    # Final save
    with open(output_path, "w") as f:
        json.dump(prices, f, indent=2)

    # Stats
    tickers_with_data = sum(1 for v in prices.values() if v)
    print(f"\nDone! {calls} calls, {tickers_with_data}/{len(companies)} tickers with price data")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
