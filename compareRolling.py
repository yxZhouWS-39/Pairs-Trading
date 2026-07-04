import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. 纯 yfinance 驱动：定义标普500核心核心池
# ==========================================
tickers = [
    # 科技与芯片
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "AMD", "QCOM", "CRM", "CSCO",
    # 消费、零售与饮料
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE",
    # 金融与银行
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # 工业、能源与医药
    "GE", "CAT", "HON", "BA", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH"
]

print(f"🚀 初始化股票池，共 {len(tickers)} 只股票。开始下载 2014-2026 历史数据...")
# 注意：因为要计算 2016 年初的滚动 63 日指标，数据必须从 2014 年开始下载以建立完整的滚动和筛选机制
raw_data = yf.download(tickers, start="2014-01-01", end="2026-06-01", auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# 清洗数据
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 数据下载清洗完成，有效股票共 {len(valid_tickers)} 只。")

# ==========================================
# 2. 利用回测前的时间（2014-2015）筛选相关系数 > 0.7 的组合
# ==========================================
print("\n🔍 正在计算 2014-2015 年间的历史皮尔逊相关系数...")
pre_train_prices = price_df.loc['2014-01-01':'2015-12-31']
train_returns = pre_train_prices.pct_change().dropna()
corr_matrix = train_returns.corr(method='pearson')

pairs = list(combinations(valid_tickers, 2))
selected_pairs = []

for s1, s2 in pairs:
    r_val = corr_matrix.loc[s1, s2]
    if r_val >= 0.7:
        selected_pairs.append((s1, s2, r_val))

print(f"🎯 成功筛选出 {len(selected_pairs)} 组在回测前相关系数 > 0.7 的股票对。")
print("开始运行动态 3 个月滚动套利模型循环测试...\n")

# ==========================================
# 3. 循环遍历所有组合，运行【滚动窗口套利回测】
# ==========================================
initial_capital = 10000.0
window = 252  # 3个月滚动窗口
m = 2      # 统一测试 1.0σ 动态阈值下的表现

backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    
    # --- 3.1 提取并计算两只股票全量时间的 Ratio 滚动指标 ---
    pair_data = pd.DataFrame(index=price_df.index)
    pair_data['A'] = price_df[stockA]
    pair_data['B'] = price_df[stockB]
    pair_data['Ratio'] = pair_data['A'] / pair_data['B']
    
    # 按照你的逻辑计算动态前3个月指标
    pair_data['Rolling_Mean'] = pair_data['Ratio'].rolling(window=window).mean()
    pair_data['Rolling_Pct_Dev'] = (pair_data['Ratio'] - pair_data['Rolling_Mean']) / pair_data['Rolling_Mean'] * 100
    pair_data['Rolling_Std'] = pair_data['Rolling_Pct_Dev'].rolling(window=window).std()
    
    # 截取真实的测试集区间 (2016 - 2026)
    test_data = pair_data.loc['2016-01-01':'2026-06-01'].copy()
    
    # --- 3.2 回测参数初始化 ---
    first_day_A = test_data['A'].iloc[0]
    first_day_B = test_data['B'].iloc[0]
    
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    trade_status = 0  # 0: 基础持仓, 1: 多A空B(75/25), -1: 空A多B(25/75)
    trade_count = 0
    
    final_portfolio_val = initial_capital
    
    # --- 3.3 逐日回测状态机循环 ---
    for date, row in test_data.iterrows():
        pct_dev = row['Rolling_Pct_Dev']
        current_std = row['Rolling_Std']
        price_A = row['A']
        price_B = row['B']
        
        # 实时计算包含两支股票持仓总价值的资产总额
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        final_portfolio_val = current_val
        
        current_threshold = m * current_std
        
        # 融入你修正后的 25% / 75% 调仓套利逻辑
        if trade_status == 0:
            # 突破正向阈值 -> 调仓为：25% A + 75% B
            if pct_dev >= current_threshold:
                trade_status = -1
                shares_A = (current_val * 0.25) / price_A
                shares_B = (current_val * 0.75) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
            # 跌破负向阈值 -> 调仓为：75% A + 25% B
            elif pct_dev <= -current_threshold:
                trade_status = 1
                shares_A = (current_val * 0.75) / price_A
                shares_B = (current_val * 0.25) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
                
        elif trade_status == -1:
            # 回归到 0 轴 -> 恢复 50% / 50% 均等持仓
            if pct_dev <= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0
                
        elif trade_status == 1:
            # 回归到 0 轴 -> 恢复 50% / 50% 均等持仓
            if pct_dev >= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0

    # --- 3.4 计算该组合的买入持有（Buy & Hold）基准最终收益率 ---
    bh_final_val = (initial_capital * 0.5 / first_day_A) * test_data['A'].iloc[-1] + \
                   (initial_capital * 0.5 / first_day_B) * test_data['B'].iloc[-1]
    
    strategy_return = (final_portfolio_val - initial_capital) / initial_capital * 100
    bh_return = (bh_final_val - initial_capital) / initial_capital * 100
    alpha = strategy_return - bh_return
    
    # 存储测试结果
    backtest_results.append({
        "Stock_1": stockA,
        "Stock_2": stockB,
        "Hist_Corr": round(corr_score, 3),
        "Strat_Return(%)": round(strategy_return, 2),
        "B&H_Return(%)": round(bh_return, 2),
        "Alpha(%)": round(alpha, 2),
        "Trades": trade_count
    })

# ==========================================
# 4. 汇总展现排行榜
# ==========================================
results_df = pd.DataFrame(backtest_results)
# 按照超额收益 (Alpha) 从高到低排序，找出最能击败市场的明星配对组合
results_df = results_df.sort_values(by="Alpha(%)", ascending=False).reset_index(drop=True)

print("============ 📊 动态滚动套利模型扫描排行榜 (2016-2026) ============")
print(results_df.to_string())