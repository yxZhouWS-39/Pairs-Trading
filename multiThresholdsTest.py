import yfinance as yf
import pandas as pd
import numpy as np
import math
from itertools import combinations

# ==========================================
# 1. Pure yfinance Driven: Define S&P 500 Core Stock Pool
# ==========================================
tickers = [
    # Tech & Chips
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "AMD", "QCOM", "CRM", "CSCO",
    # Consumer, Retail & Beverages
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE",
    # Financials & Banking
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # Industrials, Energy & Healthcare
    "GE", "CAT", "HON", "BA", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH"
]

print(f"🚀 Initializing stock pool with {len(tickers)} tickers. Starting to download full historical data from 2006 to 2026...")
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# Data cleaning
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 Data cleaning complete. Total valid tickers remaining: {len(valid_tickers)}.")

# ==========================================
# 2. Dataset Splitting (2006-2015 Train Set for correlation/metrics, 2016-2026 Test Set for live backtesting)
# ==========================================
train_prices = price_df.loc['2006-01-01':'2015-12-31'].copy()
test_prices = price_df.loc['2016-01-01':'2026-06-01'].copy()

# ==========================================
# 3. Filter Stock Pairs with Pearson Correlation Coefficient > 0.7 in [First 10 Years Train Set]
# ==========================================
print("\n🔍 Calculating historical Pearson correlation coefficients for the first 10 years (2006-2015)...")
train_returns = train_prices.pct_change().dropna()
corr_matrix = train_returns.corr(method='pearson')

pairs = list(combinations(valid_tickers, 2))
selected_pairs = []

for s1, s2 in pairs:
    r_val = corr_matrix.loc[s1, s2]
    if r_val >= 0.7:
        selected_pairs.append((s1, s2, r_val))

print(f"🎯 Successfully filtered {len(selected_pairs)} highly correlated stock pairs with Pearson R > 0.7 in the training period.")
print("Starting the [Stepped Dual-Threshold Arbitrage] backtest for each pair...\n")

# ==========================================
# 4. Encapsulate Adaptive Stepped Dual-Threshold Arbitrage Backtest Function
# ==========================================
initial_capital = 10000.0
t1 = 1.0  # Level 1: Mild imbalance trigger line
t2 = 2.0  # Level 2: Extreme imbalance scaling line

def run_stepped_backtest(df_prices, stock_A, stock_B, mean_ratio, sigma):
    """
    Runs a stepped-position arbitrage backtest tailored for dynamic stock A and stock B inputs.
    """
    threshold_1 = t1 * sigma
    threshold_2 = t2 * sigma
    
    # Dynamically build price ratio and percentage deviation series
    ratio_series = df_prices[stock_A] / df_prices[stock_B]
    pct_dev_series = (ratio_series - mean_ratio) / mean_ratio * 100
    
    first_day_A = df_prices[stock_A].iloc[0]
    first_day_B = df_prices[stock_B].iloc[0]
    
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    
    # Stepped state machine markers
    trade_status = 0 
    trade_count = 0
    portfolio_values = []
    
    for date, pct_dev in pct_dev_series.items():
        price_A = df_prices.loc[date, stock_A]
        price_B = df_prices.loc[date, stock_B]
        
        # Daily total asset valuation
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        portfolio_values.append(current_val)
        
        # Stepped position dynamic allocation logic
        if trade_status == 0:
            if pct_dev >= threshold_1 and pct_dev < threshold_2:
                trade_status = -1
                shares_A = (current_val * 0.25) / price_A
                shares_B = (current_val * 0.75) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
            elif pct_dev >= threshold_2:
                trade_status = -2
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= -threshold_1 and pct_dev > -threshold_2:
                trade_status = 1
                shares_A = (current_val * 0.75) / price_A
                shares_B = (current_val * 0.25) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= -threshold_2:
                trade_status = 2
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1

        elif trade_status == -1:
            if pct_dev >= threshold_2:
                trade_status = -2
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == -2:
            if pct_dev <= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == 1:
            if pct_dev <= -threshold_2:
                trade_status = 2
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1
            elif pct_dev >= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == 2:
            if pct_dev >= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                
    return portfolio_values, trade_count

# ==========================================
# 5. Automated Backtesting Pipeline Loop
# ==========================================
backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    # 5.1 Calculate historical statistics on the training set
    train_ratio = train_prices[stockA] / train_prices[stockB]
    mean_ratio = train_ratio.mean()
    train_pct_dev = (train_ratio - mean_ratio) / mean_ratio * 100
    std_dev = train_pct_dev.std()
    
    # 5.2 Direct application of statistics to the test set (latter 10 years) for stepped backtesting
    test_p_vals, test_trade_count = run_stepped_backtest(test_prices, stockA, stockB, mean_ratio, std_dev)
    strategy_final_return = (test_p_vals[-1] - initial_capital) / initial_capital * 100
    
    # 5.3 Calculate corresponding Buy & Hold benchmark for the test set
    test_first_A = test_prices[stockA].iloc[0]
    test_first_B = test_prices[stockB].iloc[0]
    bh_final_val = (initial_capital * 0.5 / test_first_A) * test_prices[stockA].iloc[-1] + \
                   (initial_capital * 0.5 / test_first_B) * test_prices[stockB].iloc[-1]
    bh_final_return = (bh_final_val - initial_capital) / initial_capital * 100
    
    alpha = strategy_final_return - bh_final_return
    
    backtest_results.append({
        "Stock_1": stockA,
        "Stock_2": stockB,
        "In_Sample_Corr": round(corr_score, 2),
        "Strat_Return(%)": round(strategy_final_return, 2),
        "B&H_Return(%)": round(bh_final_return, 2),
        "Alpha(%)": round(alpha, 2),
        "Trades": test_trade_count
    })

results_df = pd.DataFrame(backtest_results)
results_df = results_df.sort_values(by="Strat_Return(%)", ascending=False).reset_index(drop=True)

print("\n============ 🏆 Cross-Market High-Correlation Pairs: Stepped Arbitrage Leaderboard (2016-2026) ============")
print(results_df.to_string())

# ==========================================
# 6. [Pure Manual Implementation]: Statistical Significance Testing Module (No External Libraries)
# ==========================================
def normal_cdf(x):
    """Cumulative Distribution Function (CDF) for standard normal distribution"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def student_t_cdf_approx(t, df):
    """High-precision estimation of Student's t-distribution cumulative probability using Wallace's approximation formula"""
    if df <= 0:
        return 0.5
    z = math.sqrt(df * math.log(1.0 + (t**2) / df))
    if t < 0:
        z = -z
    return normal_cdf(z)

print("\n" + "="*50)
print("🔬 Running Statistical Significance Tests (For Stepped Trigger Strategy)...")
print("="*50)

total_pairs = len(results_df)

if total_pairs > 1:
    successful_pairs = int(np.sum(results_df['Alpha(%)'] > 0))
    win_rate = successful_pairs / total_pairs
    
    print(f"📊 Stepped Strategy Performance Overview Summary:")
    print(f"  - Total Evaluated Pairs (N): {total_pairs} pairs")
    print(f"  - Benchmark Outperforming Pairs: {successful_pairs} pairs")
    print(f"  - Strategy Win Rate (Pair-Level): {win_rate * 100:.2f}%")
    print(f"  - Average Strategy Return: {results_df['Strat_Return(%)'].mean():.2f}%")
    print(f"  - Average Buy & Hold Return: {results_df['B&H_Return(%)'].mean():.2f}%")
    print(f"  - Average Strategy Alpha: {results_df['Alpha(%)'].mean():.2f}%\n")

    # ------------------------------------------
    # Test 1: One-Tailed Proportion Z-Test
    # ------------------------------------------
    p_0 = 0.5
    se_z = math.sqrt((p_0 * (1 - p_0)) / total_pairs)
    stat_z = (win_rate - p_0) / se_z
    p_val_z = 1.0 - normal_cdf(stat_z)
    
    print(f"1️⃣ 【One-Tailed Proportion Z-Test】:")
    print(f"  - H0: Probability of strategy outperforming benchmark p <= 0.5 (Pure Luck)")
    print(f"  - Ha: Probability of strategy outperforming benchmark p > 0.5 (Genuine Strategy Edge)")
    print(f"  - Z Statistic: {stat_z:.4f}")
    print(f"  - P-Value: {p_val_z:.6f}")
    if p_val_z < 0.05:
        print(f"  📢 Conclusion: Reject H0! P-value < 0.05. The strategy's win rate is [Statistically Highly Significant].")
    else:
        print(f"  📢 Conclusion: Fail to reject H0. P-value >= 0.05. The possibility of pure luck cannot be excluded.")
        
    print("\n" + "-"*40)

    # ------------------------------------------
    # Test 2: Paired Samples t-Test
    # ------------------------------------------
    diff = results_df['Strat_Return(%)'] - results_df['B&H_Return(%)']
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    
    df_t = total_pairs - 1
    se_t = std_diff / math.sqrt(total_pairs)
    stat_t = mean_diff / se_t
    
    p_val_t = 2.0 * (1.0 - student_t_cdf_approx(abs(stat_t), df_t))
    
    print(f"2️⃣ 【Paired Samples t-Test】:")
    print(f"  - H0: Mean net Alpha generated by the stepped strategy = 0 (No difference from Buy & Hold)")
    print(f"  - Ha: Mean net Alpha generated by the stepped strategy != 0 (Significant difference exists)")
    print(f"  - Degrees of Freedom (df): {df_t}")
    print(f"  - t Statistic: {stat_t:.4f}")
    print(f"  - P-Value: {p_val_t:.6f}")
    if p_val_t < 0.05:
        print(f"  📢 Conclusion: Reject H0! P-value < 0.05. The strategy's excess returns possess [Academic-Grade Significance].")
    else:
        print(f"  📢 Conclusion: Fail to reject H0. P-value >= 0.05. Cannot prove the strategy consistently beats Buy & Hold.")
else:
    print("❌ Insufficient sample size (pairs count too low) to run statistical significance tests.")