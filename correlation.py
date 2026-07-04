import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. Stock Pool Definition Driven Purely by yfinance (Core S&P 500 Components)
# ==========================================
# Leading stocks from various S&P 500 sectors are selected here, eliminating the need to scrape Wikipedia.
tickers = [
    # Tech & Chips
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "AMD", "INTC", "QCOM", "CRM", "ORCL", "CSCO", "ADBE",
    # Consumer, Beverages & Retail
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE", "SBUX", "EL", "CL", "PM", "MO",
    # Finance & Banking
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK", "SPGI",
    # Healthcare & Pharma
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "UNH", "CVS", "TMO", "AMGN", "ISRG",
    # Industrials, Manufacturing & Aerospace
    "GE", "MMM", "CAT", "HON", "LMT", "BA", "UPS", "FDX", "DE", "EMR",
    # Energy & Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "APD", "FCX", "NUE",
    # Telecom & Utilities
    "T", "VZ", "TMUS", "NEE", "DUK", "SO", "AEP",
    # Auto & Entertainment
    "TSLA", "F", "GM", "DIS", "NFLX", "CMCSA"
]

print(f"Custom S&P 500 core stock pool initialized with {len(tickers)} tickers.")

# ==========================================
# 2. Download Historical Price Data Purely Using yfinance (2024 - 2026)
# ==========================================
start_date = "2024-06-01"
end_date = "2026-06-01"
print(f"Downloading Adjusted Close price data via yfinance from {start_date} to {end_date}...")

# yf.download will fetch and download data directly from Yahoo Finance
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# Data Cleaning: Remove dates or tickers with missing values
price_df = price_df.dropna(axis=1, how='all') # Drop entirely empty columns
price_df = price_df.ffill().bfill()            # Forward/backward fill occasional missing trading days

print(f"yfinance data download and cleaning complete. Number of tickers actually used in calculation: {price_df.shape[1]}.")

# ==========================================
# 3. Calculate Daily Returns
# ==========================================
# Calculate percentage changes, as Pearson correlation must be based on returns
returns_df = price_df.pct_change().dropna()

# ==========================================
# 4. Calculate Pearson Correlation Matrix
# ==========================================
print("Calculating the Pearson correlation matrix...")
corr_matrix = returns_df.corr(method='pearson')

# ==========================================
# 5. Exhaustive Combination and Filtering of Stock Pairs with Correlation > 0.7
# ==========================================
print("Iterating through pairwise combinations to filter high correlation pairs with coefficient >= 0.7...")
valid_tickers = corr_matrix.columns
pairs = list(combinations(valid_tickers, 2))

results = []
for stockA, stockB in pairs:
    score = corr_matrix.loc[stockA, stockB]
    if score >= 0.7:
        results.append({
            'Stock_1': stockA,
            'Stock_2': stockB,
            'Correlation': score
        })

# Convert to DataFrame and sort by correlation in descending order
high_corr_df = pd.DataFrame(results)
if not high_corr_df.empty:
    high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

# ==========================================
# 6. Output Results
# ==========================================
print(f"\nSearch complete! Found {len(high_corr_df)} pairs with a correlation coefficient greater than 0.7 in the current stock pool.\n")

if not high_corr_df.empty:
    print("--- Top 20 Stock Pairs with Highest Correlation ---")
    print(high_corr_df.head(20).to_string())
else:
    print("No stock pairs found with a correlation coefficient greater than 0.7.")