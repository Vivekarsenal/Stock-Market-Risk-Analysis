"""
=======================================================
  Project 3: Stock Market Performance & Portfolio Risk Analysis
  Author: [Your Name]
  Tools: Python, Pandas, Matplotlib, Seaborn, NumPy
  Dataset: Synthetic OHLCV Data — 10 Stocks, ~2 Years (2022–2023)
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.facecolor':   '#0F172A',
    'axes.facecolor':     '#1E293B',
    'text.color':         '#E2E8F0',
    'axes.labelcolor':    '#94A3B8',
    'xtick.color':        '#64748B',
    'ytick.color':        '#64748B',
    'axes.edgecolor':     '#334155',
    'grid.color':         '#1E293B',
})

SECTOR_COLORS = {
    'Technology': '#38BDF8',
    'Finance':    '#F59E0B',
    'Healthcare': '#4ADE80',
    'E-commerce': '#A78BFA',
    'Automotive': '#F87171',
    'Energy':     '#FB923C',
    'Retail':     '#34D399',
}
TICKER_SECTOR = {
    'AAPL':'Technology','MSFT':'Technology','JPM':'Finance','GS':'Finance',
    'JNJ':'Healthcare','PFE':'Healthcare','AMZN':'E-commerce',
    'TSLA':'Automotive','XOM':'Energy','WMT':'Retail'
}
GREEN, RED = '#4ADE80', '#F87171'
TICKERS    = list(TICKER_SECTOR.keys())

# ── 1. Load & Prepare ─────────────────────────────────────────────────────
df = pd.read_csv('data/stock_prices.csv', parse_dates=['date'])
df = df.sort_values(['ticker','date']).reset_index(drop=True)
print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Shape        : {df.shape}")
print(f"Date Range   : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Stocks       : {sorted(df['ticker'].unique())}")
print(f"Null values  : {df.isnull().sum().sum()}")
print(df.head())

# ── 2. Feature Engineering ─────────────────────────────────────────────────
df['daily_return']   = df.groupby('ticker')['close'].pct_change()
df['log_return']     = np.log(df['close'] / df.groupby('ticker')['close'].shift(1))
df['price_range']    = df['high'] - df['low']
df['30d_ma']         = df.groupby('ticker')['close'].transform(lambda x: x.rolling(30).mean())
df['7d_ma']          = df.groupby('ticker')['close'].transform(lambda x: x.rolling(7).mean())
df['volatility_30d'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(30).std() * np.sqrt(252))
df['month']          = df['date'].dt.to_period('M').astype(str)
df['year']           = df['date'].dt.year
df['quarter']        = df['date'].dt.to_period('Q').astype(str)

print("\nFeature engineering complete.")
print(df[['ticker','date','close','daily_return','30d_ma','volatility_30d']].dropna().head(10))

# ── 3. SQL-style Analysis ──────────────────────────────────────────────────
print("\n\n" + "="*60)
print("SQL-STYLE ANALYSIS")
print("="*60)

# Pivot: Close prices wide
price_pivot = df.pivot(index='date', columns='ticker', values='close').dropna()

# Q1: Total return for each stock
first_last = (df.groupby('ticker')['close']
                .agg(['first','last'])
                .assign(total_return=lambda x: (x['last']-x['first'])/x['first']*100)
                .sort_values('total_return', ascending=False))
first_last['sector'] = first_last.index.map(TICKER_SECTOR)
print("\n[Q1] Total Return (%) per Stock:")
print(first_last[['first','last','total_return','sector']].round(2))

# Q2: Annualised volatility
vol = (df.groupby('ticker')['daily_return']
         .std()
         .mul(np.sqrt(252)*100)
         .sort_values(ascending=False)
         .round(2))
print("\n[Q2] Annualised Volatility (%) per Stock:")
print(vol)

# Q3: Sharpe Ratio (risk-free rate = 4.5%)
rf_daily  = 0.045 / 252
sharpe = (df.groupby('ticker')['daily_return']
            .apply(lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(252))
            .sort_values(ascending=False)
            .round(3))
print("\n[Q3] Annualised Sharpe Ratio (rf=4.5%):")
print(sharpe)

# Q4: Max Drawdown
def max_drawdown(prices):
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return drawdown.min() * 100

mdd = (price_pivot.apply(max_drawdown)
                  .sort_values()
                  .round(2))
print("\n[Q4] Maximum Drawdown (%) per Stock:")
print(mdd)

# Q5: Correlation matrix of returns
ret_pivot = df.pivot(index='date', columns='ticker', values='daily_return').dropna()
corr_matrix = ret_pivot.corr().round(3)
print("\n[Q5] Return Correlation Matrix:")
print(corr_matrix)

# Q6: Best & worst months per stock
monthly = (df.groupby(['ticker','month'])['daily_return']
             .sum()
             .reset_index())
best_month  = monthly.loc[monthly.groupby('ticker')['daily_return'].idxmax()][['ticker','month','daily_return']]
worst_month = monthly.loc[monthly.groupby('ticker')['daily_return'].idxmin()][['ticker','month','daily_return']]
print("\n[Q6a] Best Month per Stock:")
print(best_month.set_index('ticker'))
print("\n[Q6b] Worst Month per Stock:")
print(worst_month.set_index('ticker'))

# Q7: Equal-weight portfolio
portfolio_ret = ret_pivot.mean(axis=1)
portfolio_cum = (1 + portfolio_ret).cumprod()
port_ann_ret  = portfolio_ret.mean() * 252 * 100
port_ann_vol  = portfolio_ret.std() * np.sqrt(252) * 100
port_sharpe   = (portfolio_ret.mean() - rf_daily) / portfolio_ret.std() * np.sqrt(252)
port_mdd      = max_drawdown((1 + portfolio_ret).cumprod())
print(f"\n[Q7] Equal-Weight Portfolio KPIs:")
print(f"  Annualised Return   : {port_ann_ret:.2f}%")
print(f"  Annualised Volatility: {port_ann_vol:.2f}%")
print(f"  Sharpe Ratio        : {port_sharpe:.3f}")
print(f"  Max Drawdown        : {port_mdd:.2f}%")

# ── 4. Visualizations ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#0F172A')
fig.suptitle('Stock Market Performance & Portfolio Risk Analysis',
             fontsize=24, fontweight='bold', color='#F1F5F9', y=0.98)
fig.text(0.5, 0.957, '10 Stocks · 2 Years (2022–2023) · Equal-Weight Portfolio Analysis',
         ha='center', fontsize=12, color='#64748B')

# KPI Banner
kpis = [
    ('📈 Portfolio Return',    f"{port_ann_ret:+.1f}%"),
    ('📉 Portfolio Volatility', f"{port_ann_vol:.1f}%"),
    ('⚡ Sharpe Ratio',         f"{port_sharpe:.2f}"),
    ('🔻 Max Drawdown',         f"{port_mdd:.1f}%"),
    ('🏆 Best Stock',           f"{first_last['total_return'].idxmax()}  {first_last['total_return'].max():+.1f}%"),
]
for i, (label, val) in enumerate(kpis):
    ax_k = fig.add_axes([0.01 + i*0.196, 0.908, 0.185, 0.04])
    ax_k.set_facecolor('#1E293B')
    ax_k.axis('off')
    ax_k.text(0.5, 0.75, label, ha='center', va='center', transform=ax_k.transAxes,
              fontsize=8, color='#64748B')
    color = GREEN if ('+' in val or (i == 2 and port_sharpe > 0)) else RED
    if i in [1, 3]: color = '#F59E0B'
    ax_k.text(0.5, 0.2, val, ha='center', va='center', transform=ax_k.transAxes,
              fontsize=13, fontweight='bold', color=color)

gs = gridspec.GridSpec(3, 3, figure=fig, top=0.90, bottom=0.04,
                       hspace=0.5, wspace=0.38)

# ── P1: Normalised Price (cumulative return) ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#1E293B')
norm_prices = price_pivot / price_pivot.iloc[0] * 100
for ticker in TICKERS:
    color = SECTOR_COLORS[TICKER_SECTOR[ticker]]
    ax1.plot(norm_prices.index, norm_prices[ticker],
             linewidth=1.4, label=ticker, color=color, alpha=0.85)
ax1.axhline(y=100, color='#475569', linestyle='--', linewidth=0.8, alpha=0.6)
ax1.set_ylabel('Normalised Price (Base=100)', color='#94A3B8', fontsize=9)
ax1.set_title('Cumulative Price Performance — All Stocks', fontweight='bold',
              fontsize=12, color='#F1F5F9')
ax1.legend(ncol=5, fontsize=7.5, loc='upper left',
           facecolor='#1E293B', edgecolor='#334155', labelcolor='#CBD5E1')
ax1.tick_params(colors='#64748B')

# ── P2: Total Return Bar ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1E293B')
ret_sorted = first_last['total_return'].sort_values(ascending=True)
colors_ret = [GREEN if v >= 0 else RED for v in ret_sorted]
ax2.barh(ret_sorted.index, ret_sorted.values, color=colors_ret, height=0.6, edgecolor='#0F172A')
ax2.axvline(x=0, color='#475569', linewidth=0.8)
ax2.set_xlabel('Total Return (%)', color='#94A3B8', fontsize=9)
ax2.set_title('Total Return by Stock', fontweight='bold', fontsize=11, color='#F1F5F9')
for i, (idx, val) in enumerate(ret_sorted.items()):
    ax2.text(val + (1 if val >= 0 else -1), i,
             f'{val:+.1f}%', va='center', fontsize=8,
             color=GREEN if val >= 0 else RED, ha='left' if val >= 0 else 'right')

# ── P3: Sharpe Ratio ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor('#1E293B')
sharpe_sorted = sharpe.sort_values(ascending=True)
clr3 = [GREEN if v > 0 else RED for v in sharpe_sorted]
ax3.barh(sharpe_sorted.index, sharpe_sorted.values, color=clr3, height=0.6, edgecolor='#0F172A')
ax3.axvline(x=0, color='#475569', linewidth=0.8)
ax3.axvline(x=1, color='#F59E0B', linewidth=0.8, linestyle='--', alpha=0.7, label='Sharpe=1')
ax3.set_xlabel('Sharpe Ratio', color='#94A3B8', fontsize=9)
ax3.set_title('Risk-Adjusted Return (Sharpe)', fontweight='bold', fontsize=11, color='#F1F5F9')
ax3.legend(fontsize=8, facecolor='#1E293B', edgecolor='#334155', labelcolor='#CBD5E1')
for i, (idx, val) in enumerate(sharpe_sorted.items()):
    ax3.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=8,
             color=GREEN if val > 0 else RED)

# ── P4: Volatility ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor('#1E293B')
vol_sorted = vol.sort_values(ascending=False)
clr4 = [SECTOR_COLORS[TICKER_SECTOR[t]] for t in vol_sorted.index]
ax4.bar(vol_sorted.index, vol_sorted.values, color=clr4, edgecolor='#0F172A', width=0.6)
ax4.axhline(y=vol_sorted.mean(), color='#F59E0B', linestyle='--',
            linewidth=1, alpha=0.8, label=f'Avg {vol_sorted.mean():.1f}%')
ax4.set_ylabel('Annualised Volatility (%)', color='#94A3B8', fontsize=9)
ax4.set_title('Annualised Volatility by Stock', fontweight='bold', fontsize=11, color='#F1F5F9')
ax4.tick_params(axis='x', rotation=30, labelsize=8, colors='#64748B')
ax4.legend(fontsize=8, facecolor='#1E293B', edgecolor='#334155', labelcolor='#CBD5E1')

# ── P5: Correlation Heatmap ──
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor('#1E293B')
mask = np.zeros_like(corr_matrix, dtype=bool)
sns.heatmap(corr_matrix, ax=ax5, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.4, linecolor='#0F172A',
            annot_kws={'size': 6.5, 'color': '#F1F5F9'},
            cbar_kws={'label': 'Correlation', 'shrink': 0.8})
ax5.set_title('Return Correlation Matrix', fontweight='bold', fontsize=11, color='#F1F5F9')
ax5.tick_params(axis='x', rotation=45, labelsize=7, colors='#94A3B8')
ax5.tick_params(axis='y', rotation=0, labelsize=7, colors='#94A3B8')
ax5.figure.axes[-1].yaxis.label.set_color('#94A3B8')
ax5.figure.axes[-1].tick_params(colors='#94A3B8')

# ── P6: Portfolio vs Best Stock ──
ax6 = fig.add_subplot(gs[2, :2])
ax6.set_facecolor('#1E293B')
cum_ret = (1 + ret_pivot).cumprod()
ax6.plot(cum_ret.index, portfolio_cum.values, color='#38BDF8',
         linewidth=2.5, label='Equal-Weight Portfolio', zorder=5)
best_t = first_last['total_return'].idxmax()
worst_t = first_last['total_return'].idxmin()
ax6.plot(cum_ret.index, cum_ret[best_t], color=GREEN,
         linewidth=1.4, alpha=0.8, linestyle='--', label=f'Best: {best_t}')
ax6.plot(cum_ret.index, cum_ret[worst_t], color=RED,
         linewidth=1.4, alpha=0.8, linestyle='--', label=f'Worst: {worst_t}')
ax6.fill_between(cum_ret.index, 1, portfolio_cum.values,
                 where=portfolio_cum.values >= 1, alpha=0.12, color=GREEN)
ax6.fill_between(cum_ret.index, 1, portfolio_cum.values,
                 where=portfolio_cum.values < 1, alpha=0.12, color=RED)
ax6.axhline(y=1, color='#475569', linewidth=0.8, linestyle='-')
ax6.set_ylabel('Cumulative Return (Base=1)', color='#94A3B8', fontsize=9)
ax6.set_title('Portfolio vs Best/Worst Stock — Cumulative Return', fontweight='bold',
              fontsize=12, color='#F1F5F9')
ax6.legend(fontsize=8.5, facecolor='#1E293B', edgecolor='#334155', labelcolor='#CBD5E1')
ax6.tick_params(colors='#64748B')

# ── P7: Max Drawdown ──
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor('#1E293B')
mdd_sorted = mdd.sort_values()
clr7 = [SECTOR_COLORS[TICKER_SECTOR[t]] for t in mdd_sorted.index]
ax7.barh(mdd_sorted.index, mdd_sorted.values, color=clr7, height=0.6, edgecolor='#0F172A')
ax7.set_xlabel('Max Drawdown (%)', color='#94A3B8', fontsize=9)
ax7.set_title('Maximum Drawdown by Stock', fontweight='bold', fontsize=11, color='#F1F5F9')
for i, (idx, val) in enumerate(mdd_sorted.items()):
    ax7.text(val - 0.5, i, f'{val:.1f}%', va='center', ha='right', fontsize=8, color=RED)

# Legend for sectors
sector_patches = [plt.Rectangle((0,0),1,1, color=c, label=s) for s,c in SECTOR_COLORS.items()]
fig.legend(handles=sector_patches, loc='lower center', ncol=7,
           facecolor='#1E293B', edgecolor='#334155', labelcolor='#CBD5E1',
           fontsize=8, title='Sectors', title_fontsize=8.5,
           bbox_to_anchor=(0.5, 0.01))

plt.savefig('outputs/stock_market_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0F172A')
plt.close()
print("\n✅ Dashboard saved → outputs/stock_market_dashboard.png")

# ── 5. Save Outputs ────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'sector':       [TICKER_SECTOR[t] for t in TICKERS],
    'total_return': first_last.loc[TICKERS,'total_return'].round(2),
    'annualised_volatility': vol.loc[TICKERS].round(2),
    'sharpe_ratio': sharpe.loc[TICKERS].round(3),
    'max_drawdown': mdd.loc[TICKERS].round(2),
})
summary.to_csv('outputs/stock_summary.csv')
print("✅ Summary saved → outputs/stock_summary.csv")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
