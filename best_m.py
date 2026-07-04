import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. 纯 yfinance 驱动：定义标普500核心股票池
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

print(f"🚀 初始化股票池，共 {len(tickers)} 只股票。开始下载 2006-2026 完整数据...")
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# 清洗数据：剔除历史缺失值严重的股票，并进行填充
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 数据清洗完成，有效股票共 {len(valid_tickers)} 只。")

# ==========================================
# 2. 划分数据集 (前10年用于计算相关性与参数寻优，后10年用于实战测试)
# ==========================================
train_prices = price_df.loc['2006-01-01':'2015-12-31'].copy()
test_prices = price_df.loc['2016-01-01':'2026-06-01'].copy()

# ==========================================
# 3. 在 [前10年训练集] 中筛选皮尔逊相关系数 > 0.7 的组合
# ==========================================
print("\n🔍 正在计算前10年（2006-2015）的历史皮尔逊相关系数...")
train_returns = train_prices.pct_change().dropna()
corr_matrix = train_returns.corr(method='pearson')

pairs = list(combinations(valid_tickers, 2))
selected_pairs = []

for s1, s2 in pairs:
    r_val = corr_matrix.loc[s1, s2]
    if r_val >= 0.7:
        selected_pairs.append((s1, s2, r_val))

print(f"🎯 成功筛选出 {len(selected_pairs)} 组在前10年相关系数 > 0.7 的高相关股票对。")
print("开始对每一个组合进行【前10年寻优 + 后10年测试】的嵌套循环...\n")

# ==========================================
# 4. 封装可动态适配任意股票对的通用回测函数
# ==========================================
initial_capital = 10000.0

def run_backtest(df_prices, stock_A, stock_B, multiplier, mean_ratio, sigma):
    """
    针对指定两只股票动态运行套利模型回测
    """
    current_threshold = multiplier * sigma
    
    # 动态构建两只股票的比例及偏差序列
    ratio_series = df_prices[stock_A] / df_prices[stock_B]
    pct_dev_series = (ratio_series - mean_ratio) / mean_ratio * 100
    
    first_day_A = df_prices[stock_A].iloc[0]
    first_day_B = df_prices[stock_B].iloc[0]
    
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    trade_status = 0 # 0: 50/50, 1: 全仓 A, -1: 全仓 B
    trade_count = 0
    portfolio_values = []
    
    for date, pct_dev in pct_dev_series.items():
        price_A = df_prices.loc[date, stock_A]
        price_B = df_prices.loc[date, stock_B]
        
        # 每日结算资产总价值
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        portfolio_values.append(current_val)
        
        # 全仓套利逻辑
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
# 5. 核心双重循环：穷举组合 -> 穷举参数 -> 实战回测
# ==========================================
multipliers_to_test = np.arange(0.1, 3.1, 0.1) # 参数穷举范围：0.1σ 到 3.0σ
backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    
    # --- 5.1 在训练集（前10年）计算该组合的基准统计量 ---
    train_ratio = train_prices[stockA] / train_prices[stockB]
    mean_ratio = train_ratio.mean()
    train_pct_dev = (train_ratio - mean_ratio) / mean_ratio * 100
    std_dev = train_pct_dev.std()
    
    # --- 5.2 穷举参数：寻找前10年收益率最高的那个偏差倍数 ---
    best_m = None
    best_train_return = -np.inf
    
    for m in multipliers_to_test:
        p_vals, t_count = run_backtest(train_prices, stockA, stockB, m, mean_ratio, std_dev)
        train_return = (p_vals[-1] - initial_capital) / initial_capital * 100
        
        if train_return > best_train_return:
            best_train_return = train_return
            best_m = round(m, 1)
            
    # --- 5.3 锁定最佳参数 best_m，在测试集（后10年）进行实战回测 ---
    test_p_vals, test_trade_count = run_backtest(test_prices, stockA, stockB, best_m, mean_ratio, std_dev)
    strategy_final_return = (test_p_vals[-1] - initial_capital) / initial_capital * 100
    
    # --- 5.4 计算该组合在测试集下的买入持有基准（Buy & Hold）收益率 ---
    test_first_A = test_prices[stockA].iloc[0]
    test_first_B = test_prices[stockB].iloc[0]
    bh_final_val = (initial_capital * 0.5 / test_first_A) * test_prices[stockA].iloc[-1] + \
                   (initial_capital * 0.5 / test_first_B) * test_prices[stockB].iloc[-1]
    bh_final_return = (bh_final_val - initial_capital) / initial_capital * 100
    
    # 计算阿尔法（超额收益）
    alpha = strategy_final_return - bh_final_return
    
    # 记录该股票对的最终战绩
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

# ==========================================
# 6. 汇总展现全市场扫描排行大表
# ==========================================
results_df = pd.DataFrame(backtest_results)
# 按照后10年的实战总收益率（Strat_Return(%)）从高到低进行最终大排行
results_df = results_df.sort_values(by="Strat_Return(%)", ascending=False).reset_index(drop=True)

print("\n============ 🏆 全市场高相关组合：参数自适应寻优套利排行榜 (2016-2026) ============")
print(results_df.to_string())