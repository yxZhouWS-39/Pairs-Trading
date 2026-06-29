import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 调取 20 年数据 (2006 - 2026)
print("正在调取数据...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2016-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# 2. 计算价格比率 Ratio (Coke / Pepsi)
data['Ratio'] = data['KO'] / data['PEP']
mean_ratio = data['Ratio'].mean()

# 3. 计算每日的百分比偏差 (Percentage Deviation from Mean)
# 含义：今天的比率比历史平均比率高/低了百分之几
data['Pct_Dev'] = (data['Ratio'] - mean_ratio) / mean_ratio * 100

# 🌟 【这里是新加的 3.5 步】：计算这个百分比偏差的标准差 (σ)
std_dev = data['Pct_Dev'].std()
# 4. 设定硬性阈值：（这里保留你原本的逻辑，但我们改成用 3个标准差 来定义异常）
upper_threshold = 3 * std_dev
lower_threshold = -3 * std_dev

# 5. 找出偏离超过 3个标准差 的月份/日期
divergent_days = data[(data['Pct_Dev'] > upper_threshold) | (data['Pct_Dev'] < lower_threshold)]
print(f"📊 发现共有 {len(divergent_days)} 个交易日偏离超过了 3个标准差！")

# 6. 画图
plt.figure(figsize=(14, 8))

# # 上图：绝对价格
# plt.subplot(2, 1, 1)
# plt.plot(data['KO'], label='Coke (KO)', color='red', alpha=0.8)
# plt.plot(data['PEP'], label='Pepsi (PEP)', color='blue', alpha=0.8)
# plt.title('Coke vs Pepsi - Prices', fontsize=12)
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True, alpha=0.3)

# 下图：百分比偏差及全新的统计学警戒线
plt.subplot(2, 1, 2)
plt.plot(data['Pct_Dev'], label='Percentage Deviation (%)', color='purple')
plt.axhline(0, color='gray', linestyle='--', label='Historical Average (0%)')

# 🌟 【这里修改了画线部分】：把原本固定的 20%，换成 1到4倍 的标准差线条
plt.axhline(1 * std_dev, color='green', linestyle='--', alpha=0.5, label='±1σ')
plt.axhline(-1 * std_dev, color='green', linestyle='--' , alpha=0.5)

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