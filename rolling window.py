import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Download Data (2006 - 2026)
# ==========================================
print("Downloading data for KO and PEP...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# Calculate price ratio: Ratio = KO / PEP
data['Ratio'] = data['KO'] / data['PEP']

# ==========================================
# 2. Split Test Dataset (2016 - 2026)
# ==========================================
# To calculate the "3-month rolling metrics" for 2016-01-01,
# our calculation must extend seamlessly from the end of 2015.
test_data = data.loc['2016-01-01':'2026-06-01'].copy()

# ==========================================
# 3. Calculate Daily 3-Month (Approx. 63 Trading Days) Rolling Mean and Std
# ==========================================
window = 400 # 3 months of trading days approx: 21 days/month * 3 = 63 days

# Calculate the daily rolling mean and rolling standard deviation
data['Rolling_Mean'] = data['Ratio'].rolling(window=window).mean()
# Calculate the daily percentage deviation of the ratio relative to its rolling mean
data['Rolling_Pct_Dev'] = (data['Ratio'] - data['Rolling_Mean']) / data['Rolling_Mean'] * 100
# Calculate the 3-month rolling standard deviation of this percentage deviation
data['Rolling_Std'] = data['Rolling_Pct_Dev'].rolling(window=window).std()

# Sync the calculated dynamic metrics back into the test dataset
test_data['Rolling_Mean'] = data['Rolling_Mean'].loc['2016-01-01':'2026-06-01']
test_data['Rolling_Pct_Dev'] = data['Rolling_Pct_Dev'].loc['2016-01-01':'2026-06-01']
test_data['Rolling_Std'] = data['Rolling_Std'].loc['2016-01-01':'2026-06-01']

# ==========================================
# 4. Define Multiple Dynamic Std Dev Thresholds to Compare
# ==========================================
threshold_multipliers = [0.5, 1.0, 1.5, 2.0]
results_rolling = {}

# Get the first day prices for initialization
first_day_ko = test_data['KO'].iloc[0]
first_day_pep = test_data['PEP'].iloc[0]
initial_capital = 10000.0

# 4.1 Calculate [Pure Buy & Hold Benchmark Group]
bh_values = []
for _, row in test_data.iterrows():
    bh_val = (initial_capital * 0.5 / first_day_ko) * row['KO'] + (initial_capital * 0.5 / first_day_pep) * row['PEP']
    bh_values.append(bh_val)
results_rolling['Buy & Hold'] = bh_values

# 4.2 Loop through each dynamic standard deviation threshold
for m in threshold_multipliers:
    cash = 0.0
    ko_shares = (initial_capital * 0.5) / first_day_ko
    pep_shares = (initial_capital * 0.5) / first_day_pep
    trade_status = 0 # 0: Base position, 1: Long KO Short PEP, -1: Short KO Long PEP
    portfolio_values = []
    trade_count = 0

    for date, row in test_data.iterrows():
        pct_dev = row['Rolling_Pct_Dev'] # Dynamic deviation
        current_std = row['Rolling_Std'] # Dynamic standard deviation
        ko_price = row['KO']
        pep_price = row['PEP']

        # Calculate current total assets
        current_val = cash + (ko_shares * ko_price) + (pep_shares * pep_price)
        portfolio_values.append(current_val)

        # Dynamic trigger threshold = multiply by the standard deviation specific to that day
        current_threshold = m * current_std

        # Core rotation logic
        if trade_status == 0:
            # Breakout above daily dynamic positive threshold -> Sell KO, Buy PEP
            if pct_dev >= current_threshold:
                trade_status = -1
                ko_shares = 0.0
                pep_shares = current_val / pep_price
                cash = current_val - (pep_shares * pep_price)
                trade_count += 1
            # Breakdown below daily dynamic negative threshold -> All in KO (Buy KO, Sell all PEP)
            elif pct_dev <= -current_threshold:
                trade_status = 1
                ko_shares = current_val / ko_price
                pep_shares = 0.0
                cash = current_val - (ko_shares * ko_price)
                trade_count += 1
        elif trade_status == -1:
            # Revert to 0-axis to close positions (Dynamic deviation reversion)
            if pct_dev <= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0
        elif trade_status == 1:
            # Revert to 0-axis to close positions (Dynamic deviation reversion)
            if pct_dev >= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0

    strategy_label = f"Rolling Arbitrage ({m}σ)"
    results_rolling[strategy_label] = portfolio_values
    final_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
    print(f"🔄 Dynamic 3-Month {m:>.1f}σ Group | Final Wealth: ${portfolio_values[-1]:,.2f} | Total Return: {final_return:.2f}% | Triggered Rounds: {trade_count} times")

# Print the results of the Buy & Hold group
bh_return = (bh_values[-1] - initial_capital) / initial_capital * 100
print(f"📊 Pure Buy & Hold Group | Final Wealth: ${bh_values[-1]:,.2f} | Total Return: {bh_return:.2f}%")

# ==========================================
# 5. Plot Single Panoramic Chart (Weekly Resampling)
# ==========================================
plt.figure(figsize=(14, 7))

combined_df = pd.DataFrame(results_rolling, index=test_data.index)
weekly_df = combined_df.resample('W-FRI').last()

for label in results_rolling.keys():
    if label == 'Buy & Hold':
        plt.plot(weekly_df.index, weekly_df[label], color='black', linestyle='--', linewidth=1.5, label=label, alpha=0.8)
    else:
        plt.plot(weekly_df.index, weekly_df[label], linewidth=1.3, label=label, alpha=0.9)

plt.axhline(initial_capital, color='red', linestyle=':', alpha=0.5, label='Initial Capital ($10k)')
plt.title('Comparison of 3-Month Rolling Arbitrage Strategies (2016 - 2026)', fontsize=13, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Total Portfolio Value ($)', fontsize=11)
plt.legend(loc='upper left', fontsize=10, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()