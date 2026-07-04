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
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE", "SBUX",
    # 金融与银行
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # 工业、能源与医药
    "GE", "CAT", "HON", "BA", "XOM", "CVX", "JNJ", "PFE", "MRK", "UNH"
]

print(f"🚀 初始化股票池成功，共 {len(tickers)} 只股票。开始下载 2006-2026 历史数据...")

# 2006 - 2026 完整数据下载
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# 清洗数据：剔除历史缺失值严重的股票
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 数据下载清洗完成，有效股票共 {len(valid_tickers)} 只。")

# ==========================================
# 2. 划分数据集 & 在[训练集]中筛选强相关组合
# ==========================================
train_prices = price_df.loc['2006-01-01':'2015-12-31'].copy()
test_prices = price_df.loc['2016-01-01':'2026-06-01'].copy()

# 基于训练集的价格计算每日收益率
train_returns = train_prices.pct_change().dropna()
# 计算皮尔逊相关系数矩阵
corr_matrix = train_returns.corr(method='pearson')

# 穷举找出在训练集中相关系数 > 0.7 的股票对
pairs = list(combinations(valid_tickers, 2))
selected_pairs = []

for s1, s2 in pairs:
    r_val = corr_matrix.loc[s1, s2]
    if r_val >= 0.7:
        selected_pairs.append((s1, s2, r_val))

print(f"🔍 在2006-2015训练集中，共找出了 {len(selected_pairs)} 组皮尔逊相关系数 > 0.7 的高相关组合。")
print("开始对每个组合跑套利模型循环测试...\n")

# ==========================================
# 3. 循环遍历所有强相关组合，回测套利模型
# ==========================================
initial_capital = 10000.0
# 设定套利模型的标准差触发阈值（这里统一用 1.0σ，你可以按需修改）
m = 2 

backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    
    # --- 3.1 训练集统计量计算 ---
    # 定义比率 Ratio = StockA / StockB
    train_ratio = train_prices[stockA] / train_prices[stockB]
    mean_ratio = train_ratio.mean()
    train_pct_dev = (train_ratio - mean_ratio) / mean_ratio * 100
    std_dev = train_pct_dev.std()
    current_threshold = m * std_dev
    
    # --- 3.2 测试集回测准备 ---
    test_ratio = test_prices[stockA] / test_prices[stockB]
    test_pct_dev = (test_ratio - mean_ratio) / mean_ratio * 100
    
    first_day_A = test_prices[stockA].iloc[0]
    first_day_B = test_prices[stockB].iloc[0]
    
    # 策略初始化
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    trade_status = 0 # 0: 50/50, 1: 全仓 A, -1: 全仓 B
    trade_count = 0
    
    final_portfolio_val = initial_capital
    
    # --- 3.3 逐日回测循环 ---
    for date, pct_dev in test_pct_dev.items():
        price_A = test_prices.loc[date, stockA]
        price_B = test_prices.loc[date, stockB]
        
        # 实时计算当前总资产价值（包含所持有股票全部价值）
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        final_portfolio_val = current_val # 循环结束时保留最后一天的数据
        
        # 套利模型轮动核心逻辑
        if trade_status == 0:
            # 偏差过大：A贵B便宜 -> 卖掉所有A，全仓B
            if pct_dev >= current_threshold:
                trade_status = -1
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            # 偏差过小：B贵A便宜 -> 卖掉所有B，全仓A
            elif pct_dev <= -current_threshold:
                trade_status = 1
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1
                
        elif trade_status == -1:
            # 回归 0 轴：全仓 B 结束，回归 50/50
            if pct_dev <= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0
                
        elif trade_status == 1:
            # 回归 0 轴：全仓 A 结束，回归 50/50
            if pct_dev >= 0:
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_status = 0

    # --- 3.4 计算基准 (Buy & Hold) 最终价值 ---
    bh_final_val = (initial_capital * 0.5 / first_day_A) * test_prices[stockA].iloc[-1] + \
                   (initial_capital * 0.5 / first_day_B) * test_prices[stockB].iloc[-1]
    
    # 计算收益率
    strategy_return = (final_portfolio_val - initial_capital) / initial_capital * 100
    bh_return = (bh_final_val - initial_capital) / initial_capital * 100
    alpha = strategy_return - bh_return # 超额收益
    
    # 记录该组合的结果
    backtest_results.append({
        "Stock_1": stockA,
        "Stock_2": stockB,
        "Train_Corr": corr_score,
        "Strategy_Return(%)": round(strategy_return, 2),
        "B&H_Return(%)": round(bh_return, 2),
        "Alpha(%)": round(alpha, 2),
        "Trade_Count": trade_count
    })

# ==========================================
# 4. 汇总与排行输出
# ==========================================
results_df = pd.DataFrame(backtest_results)

# 按照策略收益率从高到低排序
results_df = results_df.sort_values(by="Strategy_Return(%)", ascending=False).reset_index(drop=True)

print("============ 📊 强相关组合套利回测最终排行榜 (2016-2026) ============")
print(results_df.to_string())

# 如果你想把结果导出来看，可以取消下面这行的注释：
# results_df.to_csv("arbitrage_scan_results.csv", index=False)