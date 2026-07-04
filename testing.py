import yfinance as yf
import pandas as pd
import numpy as np
import math
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

# 清洗数据
price_df = price_df.dropna(thresh=int(len(price_df) * 0.8), axis=1)
price_df = price_df.ffill().bfill()
valid_tickers = price_df.columns.tolist()
print(f"📊 数据清洗完成，有效股票共 {len(valid_tickers)} 只。")

# ==========================================
# 2. 划分数据集
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

# ==========================================
# 4. 封装通用回测函数
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
# 5. 核心双重循环回测
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

print("\n============ 🏆 策略自适应寻优套利排行榜 (2016-2026) ============")
print(results_df.to_string())


# ==========================================
# 6. 【纯手工实现】：无外部库依赖的统计检验模块
# ==========================================
def normal_cdf(x):
    """标准正态分布的累积分布函数(CDF)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def student_t_cdf_approx(t, df):
    """
    使用 Wallace (1959) 近似公式计算学生t分布的累积分布
    大样本(df > 30)时极其精准，完全满足金融回测报告需求
    """
    if df <= 0:
        return 0.5
    # 转化为正态分布近似变量 Z
    z = math.sqrt(df * math.log(1.0 + (t**2) / df))
    if t < 0:
        z = -z
    return normal_cdf(z)

print("\n" + "="*50)
print("🔬 开始运行统计学显著性检验 (底层公式原生实现)...")
print("="*50)

total_pairs = len(results_df)

if total_pairs > 1:
    successful_pairs = int(np.sum(results_df['Alpha(%)'] > 0))
    win_rate = successful_pairs / total_pairs
    
    print(f"📊 基础统计描述:")
    print(f"  - 总测试组合数 (N): {total_pairs} 组")
    print(f"  - 成功跑赢基准的组合数: {successful_pairs} 组")
    print(f"  - 组合层面的胜率 (Win Rate): {win_rate * 100:.2f}%")
    print(f"  - 策略平均收益率: {results_df['Strat_Return(%)'].mean():.2f}%")
    print(f"  - 基准平均收益率: {results_df['B&H_Return(%)'].mean():.2f}%")
    print(f"  - 平均超额收益 (Average Alpha): {results_df['Alpha(%)'].mean():.2f}%\n")

    # ------------------------------------------
    # 检验一：单侧比例 Z 检验公式手动实现
    # ------------------------------------------
    p_0 = 0.5
    # 防止分母为0
    se_z = math.sqrt((p_0 * (1 - p_0)) / total_pairs)
    stat_z = (win_rate - p_0) / se_z
    # 单侧右尾检验 P-value
    p_val_z = 1.0 - normal_cdf(stat_z)
    
    print(f"1️⃣ 【单侧比例 Z 检验 (Proportion Z-Test)】:")
    print(f"  - 假设 H0: 策略战胜大盘的概率 p <= 0.5 (策略纯靠运气)")
    print(f"  - 假设 Ha: 策略战胜大盘的概率 p > 0.5 (策略具备真本事)")
    print(f"  - Z 统计量 (Z-Score): {stat_z:.4f}")
    print(f"  - P 值 (P-Value): {p_val_z:.6f}")
    if p_val_z < 0.05:
        print(f"  📢 结论: 拒绝原假设 H0！P值小于 0.05，策略【具有显著的获利统计优势】。")
    else:
        print(f"  📢 结论: 无法拒绝原假设 H0。P值大于 0.05，尚不能证明策略不是靠运气。")
        
    print("\n" + "-"*40)

    # ------------------------------------------
    # 检验二：配对样本 t 检验公式手动实现
    # ------------------------------------------
    # 计算配对差值
    diff = results_df['Strat_Return(%)'] - results_df['B&H_Return(%)']
    mean_diff = diff.mean()
    # 手动算样本标准差 (ddof=1)
    std_diff = diff.std(ddof=1)
    
    df_t = total_pairs - 1
    se_t = std_diff / math.sqrt(total_pairs)
    stat_t = mean_diff / se_t
    
    # 计算双尾 P-value
    p_val_t = 2.0 * (1.0 - student_t_cdf_approx(abs(stat_t), df_t))
    
    print(f"2️⃣ 【配对样本 t 检验 (Paired t-Test)】:")
    print(f"  - 假设 H0: 策略总体收益率与买入持有基准无异 (均值差 = 0)")
    print(f"  - 假设 Ha: 策略总体收益率与买入持有基准存在显著差异")
    print(f"  - 自由度 (df): {df_t}")
    print(f"  - t 统计量 (t-Statistic): {stat_t:.4f}")
    print(f"  - P 值 (P-Value): {p_val_t:.6f}")
    if p_val_t < 0.05:
        print(f"  📢 结论: 拒绝原假设 H0！P值小于 0.05，策略带来的超额回报【在统计学上是显著的】。")
    else:
        print(f"  📢 结论: 无法拒绝原假设 H0。P值大于 0.05，策略差距在统计上未达到显著水平。")
else:
    print("❌ 样本组合数过少（少于2组），无法进行统计学显著性检验。")