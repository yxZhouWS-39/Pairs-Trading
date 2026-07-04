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
# 2. 划分数据集 (2006-2015 训练集算相关性与指标，2016-2026 测试集实战)
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
print("开始对每一个组合运行【阶梯式双阈值套利】回测...\n")

# ==========================================
# 4. 封装自适应阶梯式双阈值套利回测函数
# ==========================================
initial_capital = 10000.0
t1 = 1.0  # 第一级：轻度失调触发线
t2 = 2.0  # 第二级：极度失调加码线

def run_stepped_backtest(df_prices, stock_A, stock_B, mean_ratio, sigma):
    """
    针对动态输入的股票 A 和 B 运行阶梯式仓位套利回测
    """
    threshold_1 = t1 * sigma
    threshold_2 = t2 * sigma
    
    # 动态构建价格比例与偏离值
    ratio_series = df_prices[stock_A] / df_prices[stock_B]
    pct_dev_series = (ratio_series - mean_ratio) / mean_ratio * 100
    
    first_day_A = df_prices[stock_A].iloc[0]
    first_day_B = df_prices[stock_B].iloc[0]
    
    cash = 0.0
    shares_A = (initial_capital * 0.5) / first_day_A
    shares_B = (initial_capital * 0.5) / first_day_B
    
    # 阶梯状态机标记值
    trade_status = 0 
    trade_count = 0
    portfolio_values = []
    
    for date, pct_dev in pct_dev_series.items():
        price_A = df_prices.loc[date, stock_A]
        price_B = df_prices.loc[date, stock_B]
        
        # 每日结算总资产
        current_val = cash + (shares_A * price_A) + (shares_B * price_B)
        portfolio_values.append(current_val)
        
        # 阶梯仓位动态调配逻辑
        if trade_status == 0:
            if pct_dev >= threshold_1 and pct_dev < threshold_2:
                trade_status = -1
                shares_A = (current_val * 0.25) / price_A
                shares_B = (current_val * 0.75) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
            elif pct_dev >= threshold_2:
                trade_status = -2
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= -threshold_1 and pct_dev > -threshold_2:
                trade_status = 1
                shares_A = (current_val * 0.75) / price_A
                shares_B = (current_val * 0.25) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= -threshold_2:
                trade_status = 2
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1

        elif trade_status == -1:
            if pct_dev >= threshold_2:
                trade_status = -2
                shares_A = 0.0
                shares_B = current_val / price_B
                cash = current_val - (shares_B * price_B)
                trade_count += 1
            elif pct_dev <= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == -2:
            if pct_dev <= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == 1:
            if pct_dev <= -threshold_2:
                trade_status = 2
                shares_A = current_val / price_A
                shares_B = 0.0
                cash = current_val - (shares_A * price_A)
                trade_count += 1
            elif pct_dev >= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)

        elif trade_status == 2:
            if pct_dev >= 0:
                trade_status = 0
                shares_A = (current_val * 0.5) / price_A
                shares_B = (current_val * 0.5) / price_B
                cash = current_val - (shares_A * price_A) - (shares_B * price_B)
                
    return portfolio_values, trade_count

# ==========================================
# 5. 全自动化流水线循环回测
# ==========================================
backtest_results = []

for stockA, stockB, corr_score in selected_pairs:
    # 5.1 在训练集计算历史统计量
    train_ratio = train_prices[stockA] / train_prices[stockB]
    mean_ratio = train_ratio.mean()
    train_pct_dev = (train_ratio - mean_ratio) / mean_ratio * 100
    std_dev = train_pct_dev.std()
    
    # 5.2 直接用该统计量，去跑测试集（后10年）的阶梯式实战测试
    test_p_vals, test_trade_count = run_stepped_backtest(test_prices, stockA, stockB, mean_ratio, std_dev)
    strategy_final_return = (test_p_vals[-1] - initial_capital) / initial_capital * 100
    
    # 5.3 计算测试集对应的买入持有基准（Buy & Hold）
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
        "Strat_Return(%)": round(strategy_final_return, 2),
        "B&H_Return(%)": round(bh_final_return, 2),
        "Alpha(%)": round(alpha, 2),
        "Trades": test_trade_count
    })

results_df = pd.DataFrame(backtest_results)
results_df = results_df.sort_values(by="Strat_Return(%)", ascending=False).reset_index(drop=True)

print("\n============ 🏆 全市场高相关组合：阶梯式触发套利大排行榜 (2016-2026) ============")
print(results_df.to_string())

# ==========================================
# 6. 【纯手工实现】：无外部库依赖的统计显著性检验模块
# ==========================================
def normal_cdf(x):
    """标准正态分布的累积分布函数(CDF)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def student_t_cdf_approx(t, df):
    """使用 Wallace 近似公式高精度计算 t 分布累积概率"""
    if df <= 0:
        return 0.5
    z = math.sqrt(df * math.log(1.0 + (t**2) / df))
    if t < 0:
        z = -z
    return normal_cdf(z)

print("\n" + "="*50)
print("🔬 开始运行统计学显著性检验 (针对阶梯式触发策略)...")
print("="*50)

total_pairs = len(results_df)

if total_pairs > 1:
    successful_pairs = int(np.sum(results_df['Alpha(%)'] > 0))
    win_rate = successful_pairs / total_pairs
    
    print(f"📊 阶梯策略整体表现简报:")
    print(f"  - 总筛选测试组合数 (N): {total_pairs} 组")
    print(f"  - 成功战胜基准的组合数: {successful_pairs} 组")
    print(f"  - 策略组合层胜率 (Win Rate): {win_rate * 100:.2f}%")
    print(f"  - 阶梯策略平均收益率: {results_df['Strat_Return(%)'].mean():.2f}%")
    print(f"  - 买入持有平均收益率: {results_df['B&H_Return(%)'].mean():.2f}%")
    print(f"  - 阶梯策略平均 Alpha: {results_df['Alpha(%)'].mean():.2f}%\n")

    # ------------------------------------------
    # 检验一：单侧比例 Z 检验
    # ------------------------------------------
    p_0 = 0.5
    se_z = math.sqrt((p_0 * (1 - p_0)) / total_pairs)
    stat_z = (win_rate - p_0) / se_z
    p_val_z = 1.0 - normal_cdf(stat_z)
    
    print(f"1️⃣ 【单侧比例 Z 检验 (Proportion Z-Test)】:")
    print(f"  - 假设 H0: 阶梯策略战胜大盘的概率 p <= 0.5 (全靠运气)")
    print(f"  - 假设 Ha: 阶梯策略战胜大盘的概率 p > 0.5 (策略具备真本事)")
    print(f"  - Z 统计量: {stat_z:.4f}")
    print(f"  - P 值 (P-Value): {p_val_z:.6f}")
    if p_val_z < 0.05:
        print(f"  📢 结论: 拒绝原假设 H0！P值 < 0.05，阶梯策略的胜率【在统计学上极其显著】。")
    else:
        print(f"  📢 结论: 无法拒绝原假设 H0。P值 >= 0.05，无法排除靠运气的可能性。")
        
    print("\n" + "-"*40)

    # ------------------------------------------
    # 检验二：配对样本 t 检验
    # ------------------------------------------
    diff = results_df['Strat_Return(%)'] - results_df['B&H_Return(%)']
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    
    df_t = total_pairs - 1
    se_t = std_diff / math.sqrt(total_pairs)
    stat_t = mean_diff / se_t
    
    p_val_t = 2.0 * (1.0 - student_t_cdf_approx(abs(stat_t), df_t))
    
    print(f"2️⃣ 【配对样本 t 检验 (Paired t-Test)】:")
    print(f"  - 假设 H0: 阶梯策略带来的平均净阿尔法差值 = 0 (策略和买入持有没区别)")
    print(f"  - 假设 Ha: 阶梯策略带来的平均净阿尔法差值 != 0 (存在显著差异)")
    print(f"  - 自由度 (df): {df_t}")
    print(f"  - t 统计量: {stat_t:.4f}")
    print(f"  - P 值 (P-Value): {p_val_t:.6f}")
    if p_val_t < 0.05:
        print(f"  📢 结论: 拒绝原假设 H0！P值 < 0.05，阶梯策略带来的超额回报【具备学术级显著性】。")
    else:
        print(f"  📢 结论: 无法拒绝原假设 H0。P值 >= 0.05，无法证明阶梯策略能稳赢买入持有。")
else:
    print("❌ 样本组合数过少，无法运行统计显著性检验。")