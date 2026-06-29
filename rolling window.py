import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 下载数据 (2006 - 2026)
print("正在下载 KO 和 PEP 的数据...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# 计算价格比率 Ratio = KO / PEP
data['Ratio'] = data['KO'] / data['PEP']

# 2. 划分测试集 (2016 - 2026)
# 为了计算 2016-01-01 这一天的“前3个月滚动指标”，我们的计算必须从 2015 年底的数据开始无缝延伸
test_data = data.loc['2016-01-01':'2026-06-01'].copy()

# 3. 【核心修改】：利用全量数据计算每日的前 3 个月（约 63 个交易日）滚动均值和标准差
window = 63  # 3个月的交易日大概是 21天/月 * 3 = 63天

# 计算每日对应的滚动均值和滚动标准差
data['Rolling_Mean'] = data['Ratio'].rolling(window=window).mean()
# 先计算每日比率相对于当天滚动均值的百分比偏差
data['Rolling_Pct_Dev'] = (data['Ratio'] - data['Rolling_Mean']) / data['Rolling_Mean'] * 100
# 再计算这个百分比偏差的前3个月滚动标准差
data['Rolling_Std'] = data['Rolling_Pct_Dev'].rolling(window=window).std()

# 将计算好的动态指标同步回测试集中
test_data['Rolling_Mean'] = data['Rolling_Mean'].loc['2016-01-01':'2026-06-01']
test_data['Rolling_Pct_Dev'] = data['Rolling_Pct_Dev'].loc['2016-01-01':'2026-06-01']
test_data['Rolling_Std'] = data['Rolling_Std'].loc['2016-01-01':'2026-06-01']

# 4. 定义要对比的多个动态标准差阈值
threshold_multipliers = [0.5, 1.0, 1.5, 2.0]
results_rolling = {}

# 获取第一天的价格用于初始化
first_day_ko = test_data['KO'].iloc[0]
first_day_pep = test_data['PEP'].iloc[0]
initial_capital = 10000.0

# 4.1 计算【纯买入持有对照组】
bh_values = []
for _, row in test_data.iterrows():
    bh_val = (initial_capital * 0.5 / first_day_ko) * row['KO'] + (initial_capital * 0.5 / first_day_pep) * row['PEP']
    bh_values.append(bh_val)
results_rolling['Buy & Hold'] = bh_values

# 4.2 循环遍历每一个动态标准差阈值
for m in threshold_multipliers:
    cash = 0.0
    ko_shares = (initial_capital * 0.5) / first_day_ko
    pep_shares = (initial_capital * 0.5) / first_day_pep
    trade_status = 0 # 0: 基础持仓, 1: 多KO空PEP, -1: 空KO多PEP
    
    portfolio_values = []
    trade_count = 0
    
    for date, row in test_data.iterrows():
        pct_dev = row['Rolling_Pct_Dev']   # 动态偏差
        current_std = row['Rolling_Std']   # 动态标准差
        ko_price = row['KO']
        pep_price = row['PEP']
        
        # 计算当前总资产
        current_val = cash + (ko_shares * ko_price) + (pep_shares * pep_price)
        portfolio_values.append(current_val)
        
        # 动态触发阈值 = 乘以当天特有的标准差
        current_threshold = m * current_std
        
        # 核心轮次逻辑
        if trade_status == 0:
            # 突破当日动态正向阈值 -> 卖KO买PEP
            if pct_dev >= current_threshold:
                trade_status = -1
                ko_shares = (current_val * 0.25) / ko_price
                pep_shares = (current_val * 0.75) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_count += 1
            # 跌破当日动态负向阈值 -> 买KO卖PEP
            elif pct_dev <= -current_threshold:
                trade_status = 1
                ko_shares = (current_val * 0.75) / ko_price
                pep_shares = (current_val * 0.25) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_count += 1
                
        elif trade_status == -1:
            # 回归到 0 轴平仓 (动态偏差回归)
            if pct_dev <= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0
                
        elif trade_status == 1:
            # 回归到 0 轴平仓 (动态偏差回归)
            if pct_dev >= 0:
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0
                
    strategy_label = f"Rolling Arbitrage ({m}σ)"
    results_rolling[strategy_label] = portfolio_values
    final_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
    print(f"🔄 动态3个月 {m:>.1f}σ 组 | 最终资产: ${portfolio_values[-1]:,.2f} | 总收益率: {final_return:.2f}% | 触发轮次: {trade_count} 次")

# 打印买入持有组的结果
bh_return = (bh_values[-1] - initial_capital) / initial_capital * 100
print(f"📊 纯买入持有组       | 最终资产: ${bh_values[-1]:,.2f} | 总收益率: {bh_return:.2f}%")

# ==========================================
# 5. 绘制全新单张全景图表（周度降采样）
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