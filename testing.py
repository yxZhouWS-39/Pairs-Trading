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
    # Consumer, Retail & Beverage
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE",
    # Finance & Banking
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # Industrials, Energy & Healthcare
    "GE", "CAT", "HON", "BA", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH"
]

print(f"🚀 Initializing stock pool with {len(tickers)} stocks. Starting download for full 2006-2026 data...")
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# Data Cleaning
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 Data cleaning complete. Total valid stocks: {len(valid_tickers)}.")

# ==========================================
# 2. Split Dataset
# ==========================================
train_prices = price_df.loc['2006-01-01':'2015-12-31'].copy()
test_prices = price_df.loc['2016-01-01':'2026-06-01'].copy()

# ==========================================
# 3. Filter Pairs with Pearson Correlation > 0.7 in [First 10 Years Training Set]
# ==========================================
print("\n🔍 Calculating historical Pearson correlation for the first 10 years (2006-2015)...")
train_returns = train_prices.pct_change().dropna()
corr_matrix = train_returns.corr(method='pearson')

pairs = list(combinations(valid_tickers, 2))
selected_pairs = []

for s1, s2 in pairs:
    r_val = corr_matrix.loc[s1, s2]
    if r_val >= 0.7:
        selected_pairs.append((s1, s2, r_val))

print(f"🎯 Successfully screened {len(selected_pairs)} highly correlated stock pairs with correlation coefficient > 0.7 in the training set.")

# ==========================================
# 4. Encapsulate General Backtesting Function
# ==========================================
initial_capital = 10000.0

def run_backtest(df_prices, stock_A, stock_B, multiplier, mean_ratio, sigma):
    current_threshold = multiplier * sigma
    ratio_series = df_prices[stock_A] / df_prices[stock_B]
    pct_dev_series = (ratio_series - mean_ratio) / mean_ratio * 100
    
    first_day_A = df_prices[stock_A].iloc[0]
    first_day_B = df_prices[stock_B].iloc[0]
    
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    trade_status = 0 
    trade_count = 0
    portfolio_values = []
    
    for date, pct_dev in pct_dev_series.items():
        price_A = df_prices.loc[date, stock_A]
        price_B = df_prices.loc[date, stock_B]
        
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        portfolio_values.append(current_val)
        
        if trade_status == 0:
            if pct_dev >= current_threshold:
                trade_status = -1
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= -current_threshold:
                trade_status = 1
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1
        elif trade_status == -1:
            if pct_dev <= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0
        elif trade_status == 1:
            if pct_dev >= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0
                
    return portfolio_values, trade_count

# ==========================================
# 5. Core Double-Loop Backtesting
# ==========================================
multipliers_to_test = np.arange(0.1, 3.1, 0.1) 
backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    train_ratio = train_prices[stockA] / train_prices[stockB]
    mean_ratio = train_ratio.mean()
    train_pct_dev = (train_ratio - mean_ratio) / mean_ratio * 100
    std_dev = train_pct_dev.std()
    
    best_m = None
    best_train_return = -np.inf
    for m in multipliers_to_test:
        p_vals, t_count = run_backtest(train_prices, stockA, stockB, m, mean_ratio, std_dev)
        train_return = (p_vals[-1] - initial_capital) / initial_capital * 100
        if train_return > best_train_return:
            best_train_return = train_return
            best_m = round(m, 1)
            
    test_p_vals, test_trade_count = run_backtest(test_prices, stockA, stockB, best_m, mean_ratio, std_dev)
    strategy_final_return = (test_p_vals[-1] - initial_capital) / initial_capital * 100
    
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
        "Best_Sigma_m": best_m,
        "Strat_Return(%)": round(strategy_final_return, 2),
        "B&H_Return(%)": round(bh_final_return, 2),
        "Alpha(%)": round(alpha, 2),
        "Trades": test_trade_count
    })

results_df = pd.DataFrame(backtest_results)
results_df = results_df.sort_values(by="Strat_Return(%)", ascending=False).reset_index(drop=True)

print("\n============ 🏆 Adaptive Optimization Strategy Arbitrage Leaderboard (2016-2026) ============")
print(results_df.to_string())


# ==========================================
# 6. [Pure Manual Implementation]: Statistical Hypothesis Testing Module without External Libraries
# ==========================================
def normal_cdf(x):
    """Cumulative Distribution Function (CDF) for standard normal distribution"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def student_t_cdf_approx(t, df):
    """
    Approximates the Student's t-distribution CDF using Wallace (1959) formula.
    Extremely accurate for large samples (df > 30), fully meeting requirements for financial backtesting reports.
    """
    if df <= 0:
        return 0.5
    # Convert to normal approximation variable Z
    z = math.sqrt(df * math.log(1.0 + (t**2) / df))
    if t < 0:
        z = -z
    return normal_cdf(z)

print("\n" + "="*50)
print("🔬 Running Statistical Significance Test (Native Formula Implementation)...")
print("="*50)

total_pairs = len(results_df)

if total_pairs > 1:
    successful_pairs = int(np.sum(results_df['Alpha(%)'] > 0))
    win_rate = successful_pairs / total_pairs
    
    print(f"📊 Basic Descriptive Statistics:")
    print(f"  - Total Tested Pairs (N): {total_pairs} pairs")
    print(f"  - Number of Pairs Outperforming Benchmark: {successful_pairs} pairs")
    print(f"  - Pair-level Win Rate: {win_rate * 100:.2f}%")
    print(f"  - Average Strategy Return: {results_df['Strat_Return(%)'].mean():.2f}%")
    print(f"  - Average Benchmark Return: {results_df['B&H_Return(%)'].mean():.2f}%")
    print(f"  - Average Excess Return (Average Alpha): {results_df['Alpha(%)'].mean():.2f}%\n")

    # ------------------------------------------
    # Test 1: Manual Implementation of One-Sided Proportion Z-Test Formula
    # ------------------------------------------
    p_0 = 0.5
    # Prevent division by zero
    se_z = math.sqrt((p_0 * (1 - p_0)) / total_pairs)
    stat_z = (win_rate - p_0) / se_z
    # One-sided right-tailed P-value
    p_val_z = 1.0 - normal_cdf(stat_z)
    
    print(f"1️⃣ [One-Sided Proportion Z-Test]:")
    print(f"  - Hypothesis H0: Probability of outperforming the market p <= 0.5 (Strategy relies purely on luck)")
    print(f"  - Hypothesis Ha: Probability of outperforming the market p > 0.5 (Strategy possesses genuine edge)")
    print(f"  - Z-Score: {stat_z:.4f}")
    print(f"  - P-Value: {p_val_z:.6f}")
    if p_val_z < 0.05:
        print(f"  📢 Conclusion: Reject the null hypothesis H0! P-value is less than 0.05, the strategy has a [significant statistical advantage in profitability].")
    else:
        print(f"  📢 Conclusion: Fail to reject the null hypothesis H0. P-value is greater than 0.05, cannot prove the strategy is not driven by luck.")
        
    print("\n" + "-"*40)

    # ------------------------------------------
    # Test 2: Manual Implementation of Paired Samples t-Test Formula
    # ------------------------------------------
    # Calculate paired differences
    diff = results_df['Strat_Return(%)'] - results_df['B&H_Return(%)']
    mean_diff = diff.mean()
    # Manually calculate sample standard deviation (ddof=1)
    std_diff = diff.std(ddof=1)
    
    df_t = total_pairs - 1
    se_t = std_diff / math.sqrt(total_pairs)
    stat_t = mean_diff / se_t
    
    # Calculate two-tailed P-value
    p_val_t = 2.0 * (1.0 - student_t_cdf_approx(abs(stat_t), df_t))
    
    print(f"2️⃣ [Paired Samples t-Test]:")
    print(f"  - Hypothesis H0: The overall strategy return is identical to the Buy & Hold benchmark (Mean difference = 0)")
    print(f"  - Hypothesis Ha: The overall strategy return is significantly different from the Buy & Hold benchmark")
    print(f"  - Degrees of Freedom (df): {df_t}")
    print(f"  - t-Statistic: {stat_t:.4f}")
    print(f"  - P-Value: {p_val_t:.6f}")
    if p_val_t < 0.05:
        print(f"  📢 Conclusion: Reject the null hypothesis H0! P-value is less than 0.05, the excess returns generated by the strategy are [statistically significant].")
    else:
        print(f"  📢 Conclusion: Fail to reject the null hypothesis H0. P-value is greater than 0.05, the performance gap is not statistically significant.")
else:
    print("❌ Sample size is too small (less than 2 pairs) to perform statistical significance tests.")