import yfinance as yf
import pandas as pd

# ====================================================
# 1. 调取 20 年历史数据 (2006 - 2026)
# ====================================================
print("正在调取数据计算纯持有(Buy & Hold)收益...")
tickers = ["KO", "PEP"]
raw_data = yf.download(tickers, start="2006-01-01", end="2026-06-01", auto_adjust=False)
data = raw_data['Adj Close'].copy()

# ====================================================
# 2. 模拟 10,000 美元平分买入并死拿
# ====================================================
initial_capital = 10000.0
allocated_capital = initial_capital / 2  # 每只股票分 5000 美元

# 获取 2006 年第一天的两只股票价格
start_ko_price = data['KO'].iloc[0]
start_pep_price = data['PEP'].iloc[0]

# 计算第一天你可以买入多少股（买完就不动了）
ko_shares = allocated_capital / start_ko_price
pep_shares = allocated_capital / start_pep_price

# 获取 2026 年最后一天的两只股票价格
end_ko_price = data['KO'].iloc[-1]
end_pep_price = data['PEP'].iloc[-1]

# 计算今天这些股票值多少钱
final_ko_value = ko_shares * end_ko_price
final_pep_value = pep_shares * end_pep_price

# 汇总总资产
final_total_value = final_ko_value + final_pep_value
total_return = (final_total_value - initial_capital) / initial_capital * 100

# ====================================================
# 3. 打印单纯的持有收益报告
# ====================================================
print("\n" + "="*40 + " 📈 纯持有(Buy & Hold) 最终收益报告 " + "="*40)
print(f"🗓️ 模拟起点 (2006年年初价格): KO = ${start_ko_price:.2f}, PEP = ${start_pep_price:.2f}")
print(f"🛒 初始本金 $10,000 分配: 分别买入 {ko_shares:.2f} 股 KO 和 {pep_shares:.2f} 股 PEP")
print("-"*100)
print(f"🗓️ 模拟终点 (2026年最新价格): KO = ${end_ko_price:.2f}, PEP = ${end_pep_price:.2f}")
print(f"💰 20年后你的可口可乐股票价值: ${final_ko_value:,.2f}")
print(f"💰 20年后你的百事可乐股票价值: ${final_pep_value:,.2f}")
print("-"*100)
print(f"🚀 【最终总资产】: ${final_total_value:,.2f}")
print(f"📈 【纯持有总收益率】: {total_return:.2f}%")
print("="*100)