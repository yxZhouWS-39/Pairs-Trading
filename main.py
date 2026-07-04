import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Fetch 20 years of data (2006 - 2026)
print("Fetching data...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2016-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# 2. Calculate the price ratio (Coke / Pepsi)
data['Ratio'] = data['KO'] / data['PEP']
mean_ratio = data['Ratio'].mean()

# 3. Calculate daily percentage deviation (Percentage Deviation from Mean)
# Meaning: By what percentage is today's ratio higher/lower than the historical average ratio
data['Pct_Dev'] = (data['Ratio'] - mean_ratio) / mean_ratio * 100

# 🌟 [Newly added Step 3.5]: Calculate the standard deviation (σ) of this percentage deviation
std_dev = data['Pct_Dev'].std()

# 4. Set hard thresholds: (Keeping your original logic, but changing to use 3 standard deviations to define anomalies)
upper_threshold = 3 * std_dev
lower_threshold = -3 * std_dev

# 5. Identify months/dates where deviation exceeds 3 standard deviations
divergent_days = data[(data['Pct_Dev'] > upper_threshold) | (data['Pct_Dev'] < lower_threshold)]
print(f"📊 Found a total of {len(divergent_days)} trading days deviating by more than 3 standard deviations!")

# 6. Plotting
plt.figure(figsize=(14, 8))

# # Top Plot: Absolute Prices
# plt.subplot(2, 1, 1)
# plt.plot(data['KO'], label='Coke (KO)', color='red', alpha=0.8)
# plt.plot(data['PEP'], label='Pepsi (PEP)', color='blue', alpha=0.8)
# plt.title('Coke vs Pepsi - Prices', fontsize=12)
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True, alpha=0.3)

# Bottom Plot: Percentage deviation and the brand-new statistical warning lines
plt.subplot(2, 1, 2)
plt.plot(data['Pct_Dev'], label='Percentage Deviation (%)', color='purple')
plt.axhline(0, color='gray', linestyle='--', label='Historical Average (0%)')

# 🌟 [Modified line-plotting section]: Replacing the original fixed 20% with 1 to 4 times standard deviation lines
plt.axhline(1 * std_dev, color='green', linestyle='--', alpha=0.5, label='±1σ')
plt.axhline(-1 * std_dev, color='green', linestyle='--', alpha=0.5)

plt.axhline(2 * std_dev, color='orange', linestyle='--', alpha=0.7, label='±2σ')
plt.axhline(-2 * std_dev, color='orange', linestyle='--', alpha=0.7)

plt.axhline(3 * std_dev, color='blue', linestyle='-.', alpha=0.8, label='±3σ')
plt.axhline(-3 * std_dev, color='blue', linestyle='-.', alpha=0.8)

plt.axhline(4 * std_dev, color='red', linestyle=':', alpha=0.9, label='±4σ')
plt.axhline(-4 * std_dev, color='red', linestyle=':', alpha=0.9)

plt.title('Percentage Deviation from Average Ratio with Sigma Bands', fontsize=12)
plt.ylabel('Deviation (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()