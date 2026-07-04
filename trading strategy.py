import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 下载数据 (2006 - 2026)
print("正在下载 KO 和 PEP 的数据...")
tickers = ["KO", "PEP"]
# 注意：2026年数据请确保能获取到
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# 计算价格比例 Ratio = KO / PEP
data['Ratio'] = data['KO'] / data['PEP']

# 2. 划分数据集
train_data = data.loc['2006-01-01':'2015-12-31'].copy()
test_data = data.loc['2016-01-01':'2026-06-01'].copy()

# 3. 使用 [训练集] 计算历史指标
mean_ratio = train_data['Ratio'].mean()
train_pct_dev = (train_data['Ratio'] - mean_ratio) / mean_ratio * 100
std_dev = train_pct_dev.std()

print(f"\n--- 训练集 (2006-2016) 统计指标 ---")
print(f"历史平均比例 (Mean Ratio): {mean_ratio:.4f}")
print(f"偏差值的标准差 (Sigma): {std_dev:.2f}%\n")

# 4. 计算 [测试集] 上的每日百分比偏差
test_data['Pct_Dev'] = (test_data['Ratio'] - mean_ratio) / mean_ratio * 100

# 5. 定义多个标准差阈值进行对比
threshold_multipliers = [0.5, 1.0, 1.5, 2.0]
results = {}

# 获取第一天价格用于初始化
first_day_ko = test_data['KO'].iloc[0]
first_day_pep = test_data['PEP'].iloc[0]
initial_capital = 10000.0

# 5.1 计算 [买入持有 (Buy & Hold) 基准] 的每日价值
bh_values = []
for _, row in test_data.iterrows():
    # 初始10000刀，一半买KO，一半买PEP，之后一直持有
    bh_val = (initial_capital * 0.5 / first_day_ko) * row['KO'] + (initial_capital * 0.5 / first_day_pep) * row['PEP']
    bh_values.append(bh_val)
results['Buy & Hold'] = bh_values

# 5.2 循环每个标准差阈值组合进行独立回测
for m in threshold_multipliers:
    current_threshold = m * std_dev
    
    # 策略初始化：一开始10000刀，一半买KO，一半买PEP
    cash = 0.0
    ko_shares = (initial_capital * 0.5) / first_day_ko
    pep_shares = (initial_capital * 0.5) / first_day_pep
    
    # trade_status 状态机: 
    # 0: 基础仓位 (50/50), 1: 全仓 KO (100/0), -1: 全仓 PEP (0/100)
    trade_status = 0 
    
    portfolio_values = []
    trade_count = 0 
    
    for date, row in test_data.iterrows():
        pct_dev = row['Pct_Dev']
        ko_price = row['KO']
        pep_price = row['PEP']
        
        # 实时计算当前总资产：现金 + 持有KO价值 + 持有PEP价值
        current_val = cash + (ko_shares * ko_price) + (pep_shares * pep_price)
        portfolio_values.append(current_val)
        
        # 核心全仓轮动逻辑
        if trade_status == 0:
            # 1. 偏差值过大 -> 说明 KO 相对贵了，PEP 便宜了：卖掉所有 KO，全仓 PEP
            if pct_dev >= current_threshold:
                trade_status = -1
                ko_shares = 0.0
                pep_shares = current_val / pep_price
                cash = current_val - (pep_shares * pep_price)
                trade_count += 1
            # 2. 偏差值负数过于小 -> 说明 PEP 相对贵了，KO 便宜了：卖掉所有 PEP，全仓 KO
            elif pct_dev <= -current_threshold:
                trade_status = 1
                ko_shares = current_val / ko_price
                pep_shares = 0.0
                cash = current_val - (ko_shares * ko_price)
                trade_count += 1
                
        elif trade_status == -1:
            # 之前全仓了 PEP，现在等待偏差值回归 0 (从正数跌回 <= 0)
            if pct_dev <= 0:
                # 结束交易：用全部资产重新按 50% / 50% 比例购买 KO 和 PEP
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0 # 重新回到等待触发状态
                
        elif trade_status == 1:
            # 之前全仓了 KO，现在等待偏差值回归 0 (从负数涨回 >= 0)
            if pct_dev >= 0:
                # 结束交易：用全部资产重新按 50% / 50% 比例购买 KO 和 PEP
                ko_shares = (current_val * 0.5) / ko_price
                pep_shares = (current_val * 0.5) / pep_price
                cash = current_val - (ko_shares * ko_price) - (pep_shares * pep_price)
                trade_status = 0 # 重新回到等待触发状态
                
    strategy_label = f"Arbitrage ({m}σ)"
    results[strategy_label] = portfolio_values
    
    # 结尾计算总资产和总收益率（用最后一天包含全部股票价值的 portfolio_values[-1]）
    final_portfolio_value = portfolio_values[-1]
    final_return = (final_portfolio_value - initial_capital) / initial_capital * 100
    print(f"📈 策略 {m:>.1f}σ 组 | 最终总资产: ${final_portfolio_value:,.2f} | 总收益率: {final_return:.2f}% | 触发交易次数: {trade_count} 次")

# 打印买入持有基准的结果
bh_final_value = bh_values[-1]
bh_return = (bh_final_value - initial_capital) / initial_capital * 100
print(f"📊 买入持有基准  | 最终总资产: ${bh_final_value:,.2f} | 总收益率: {bh_return:.2f}%")

# ==========================================
# 6. 绘图对比
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
