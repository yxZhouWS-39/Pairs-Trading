import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 1. Download Data (2006 - 2026)
print("Downloading data for KO and PEP...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# Calculate the price ratio: Ratio = KO / PEP
data['Ratio'] = data['KO'] / data['PEP']

# 2. Split Dataset
train_data = data.loc['2006-01-01':'2015-12-31'].copy()
test_data = data.loc['2016-01-01':'2026-06-01'].copy()

# 3. Calculate Historical Metrics Using [Training Set]
mean_ratio = train_data['Ratio'].mean()
train_pct_dev = (train_data['Ratio'] - mean_ratio) / mean_ratio * 100
std_dev = train_pct_dev.std()

print(f"\n--- Training Set (2006-2016) Statistics ---")
print(f"Historical Mean Ratio: {mean_ratio:.4f}")
print(f"Standard Deviation of Deviation (Sigma): {std_dev:.2f}%\n")

# ==========================================
# New: Perform ADF Stationarity Test on Training Set Percentage Deviation Series
# ==========================================
print("Performing ADF stationarity test on training set percentage deviation (train_pct_dev)...")
adf_result = adfuller(train_pct_dev.dropna())

print(f"   ADF Test Statistic: {adf_result[0]:.4f}")
print(f"   p-value: {adf_result[1]:.4f}")
print(f"   Critical Value at 5% Significance Level: {adf_result[4]['5%']:.4f}")

# Double verification: p-value < 0.05 AND Test Statistic < 5% Critical Value
if adf_result[1] < 0.05 and adf_result[0] < adf_result[4]['5%']:
    print("[ADF Conclusion]: Reject the null hypothesis. The percentage deviation series is [STATIONARY].")
    print("                 This implies the price ratio exhibits mean-reverting properties, and pairs trading can be safely executed.")
else:
    print("[ADF Conclusion]: Fail to reject the null hypothesis. The percentage deviation series is [NON-STATIONARY].")
    print("                 ⚠️ WARNING: The series may contain a trend or random walk. Pairs trading faces risks of non-mean reversion.")
print("-" * 60 + "\n")


# 4. Calculate Daily Percentage Deviation on [Test Set]
test_data['Pct_Dev'] = (test_data['Ratio'] - mean_ratio) / mean_ratio * 100

# 5. Define Multiple Standard Deviation Thresholds for Comparison
threshold_multipliers = [0.5, 1.0, 1.5, 2.0]
results = {}

# Get the first day's prices for initialization
first_day_ko = test_data['KO'].iloc[0]
first_day_pep = test_data['PEP'].iloc[0]
initial_capital = 10000.0

# 5.1 Calculate Daily Value of the [Buy & Hold Benchmark]
bh_values = []
for _, row in test_data.iterrows():
    bh_val = (initial_capital * 0.5 / first_day_ko) * row['KO'] + (initial_capital * 0.5 / first_day_pep) * row['PEP']
    bh_values.append(bh_val)
results['Buy & Hold'] = bh_values

# 5.2 Loop through each standard deviation threshold combination for independent backtesting
for m in threshold_multipliers:
    current_threshold = m * std_dev
    
    # Strategy initialization
    cash = 0.0
    ko_shares = (initial_capital * 0.5) / first_day_ko
    pep_shares = (initial_capital * 0.5) / first_day_pep
    
    # trade_status state machine
    trade_status = 0 
    
    portfolio_values = []
    trade_count = 0 
    
    for date, row in test_data.iterrows():
        pct_dev = row['Pct_Dev']
        ko_price = row['KO']
        pep_price = row['PEP']
        
        current_val = cash + (ko_shares * ko_price) + (pep_shares * pep_price)
        portfolio_values.append(current_val)
        
        # Core all-in rotation logic
        if trade_status == 0:
            if pct_dev >= current_threshold:
                trade_status = -1
                ko_shares = 0.0
                pep_shares = current_val / pep_price
                cash = current_val - (pep_shares * pep_price)
                trade_count += 1
            elif pct_dev <= -current_threshold:
                trade_status = 1
                ko_shares = current_val / ko_price
                pep_shares = 0.0
                cash = current_val - (ko_shares * ko_price)
                trade_count += 1
                
        elif trade_status == -1:
            if pct_dev <= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0
                
        elif trade_status == 1:
            if pct_dev >= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0
                
    strategy_label = f"Arbitrage ({m}σ)"
    results[strategy_label] = portfolio_values
    
    final_portfolio_value = portfolio_values[-1]
    final_return = (final_portfolio_value - initial_capital) / initial_capital * 100
    print(f"📈 Strategy {m:>.1f}σ | Final Value: ${final_portfolio_value:,.2f} | Total Return: {final_return:.2f}% | Trades Triggered: {trade_count} times")

# Print results for the Buy & Hold benchmark
bh_final_value = bh_values[-1]
bh_return = (bh_final_value - initial_capital) / initial_capital * 100
print(f"📊 Buy & Hold Bm  | Final Value: ${bh_final_value:,.2f} | Total Return: {bh_return:.2f}%")

# ==========================================
# 6. Plotting and Comparison
# ==========================================
plt.figure(figsize=(14, 7))

combined_df = pd.DataFrame(results, index=test_data.index)
weekly_df = combined_df.resample('W-FRI').last()

for label in results.keys():
    if label == 'Buy & Hold':
        plt.plot(weekly_df.index, weekly_df[label], color='black', linestyle='--', linewidth=1.5, label=label, alpha=0.8)
    else:
        plt.plot(weekly_df.index, weekly_df[label], linewidth=1.3, label=label, alpha=0.9)

plt.axhline(initial_capital, color='red', linestyle=':', alpha=0.5, label='Initial Capital ($10k)')
plt.title('Comparison of Total-In Arbitrage Strategies with Different σ Thresholds (2016 - 2026)', fontsize=13, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Total Portfolio Value ($)', fontsize=11)
plt.legend(loc='upper left', fontsize=10, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()