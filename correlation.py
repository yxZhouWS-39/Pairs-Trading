import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ==========================================
# 1. 纯粹由 yfinance 驱动的股票池定义 (标普500核心成分股)
# ==========================================
# 这里精选了标普500各板块的龙头，完全不需要从维基百科爬取
tickers = [
    # 科技与芯片
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "AMD", "INTC", "QCOM", "CRM", "ORCL", "CSCO", "ADBE",
    # 消费、饮料与零售
    "KO", "PEP", "WMT", "COST", "PG", "HD", "MCD", "NKE", "SBUX", "EL", "CL", "PM", "MO",
    # 金融与银行
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK", "SPGI",
    # 医药与健康
    "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "UNH", "CVS", "TMO", "AMGN", "ISRG",
    # 工业、制造与航空
    "GE", "MMM", "CAT", "HON", "LMT", "BA", "UPS", "FDX", "DE", "EMR",
    # 能源与材料
    "XOM", "CVX", "COP", "SLB", "EOG", "LIN", "APD", "FCX", "NUE",
    # 电信与公用事业
    "T", "VZ", "TMUS", "NEE", "DUK", "SO", "AEP",
    # 汽车与娱乐
    "TSLA", "F", "GM", "DIS", "NFLX", "CMCSA"
]

print(f"自定义标普500核心股票池初始化完成，共 {len(tickers)} 只股票。")

# ==========================================
# 2. 纯粹使用 yfinance 下载历史价格数据 (2024 - 2026)
# ==========================================
start_date = "2024-06-01"
end_date = "2026-06-01"
print(f"正在通过 yfinance 下载从 {start_date} 到 {end_date} 的收盘价数据...")

# yf.download 会直接向 Yahoo Finance 发起请求并下载数据
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
price_df = raw_data['Adj Close'].copy()

# 清洗数据：剔除有缺失值的日期或股票
price_df = price_df.dropna(axis=1, how='all') # 剔除全空的列
price_df = price_df.ffill().bfill()            # 前向/后向填充个别交易日空缺

print(f"yfinance 数据下载并清洗完毕。实际参与计算的股票数量: {price_df.shape[1]} 只。")

# ==========================================
# 3. 计算每日收益率 (Returns)
# ==========================================
# 计算百分比变动，因为皮尔逊相关性必须基于收益率
returns_df = price_df.pct_change().dropna()

# ==========================================
# 4. 计算皮尔逊相关系数矩阵
# ==========================================
print("正在计算皮尔逊相关系数大表...")
corr_matrix = returns_df.corr(method='pearson')

# ==========================================
# 5. 穷举组合并筛选相关系数 > 0.7 的股票对
# ==========================================
print("正在穷举两两组合，筛选相关系数 >= 0.7 的高相关组合...")
valid_tickers = corr_matrix.columns
pairs = list(combinations(valid_tickers, 2))

results = []
for stockA, stockB in pairs:
    score = corr_matrix.loc[stockA, stockB]
    if score >= 0.7:
        results.append({
            'Stock_1': stockA,
            'Stock_2': stockB,
            'Correlation': score
        })

# 转化为 DataFrame 并按相关性从高到低排序
high_corr_df = pd.DataFrame(results)
if not high_corr_df.empty:
    high_corr_df = high_corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

# ==========================================
# 6. 输出结果
# ==========================================
print(f"\n寻找完毕！在当前股票池中，共找到 {len(high_corr_df)} 组相关系数大于 0.7 的组合。\n")

if not high_corr_df.empty:
    print("--- 相关性最高的前 20 组股票对 ---")
    print(high_corr_df.head(20).to_string())
else:
    print("未找到相关系数大于 0.7 的股票对。")