#!/usr/bin/env python3
"""addendum_temporal.py — 背离v6 种子实验：信号时间分布深挖（纯描述性统计，零拟合、零筛选决策）

数据源（冻结，只读）：
- experiments/divergence_seed_trial_history/trades_seed.parquet（事件级分析框 = variant=='v1' 且 H==20，96577 行）
- stock_data/daily/000001.SH.parquet（市场日历 + 指数 close，1993-10-08 ~ 2026-08-31，8000 交易日）

输出：
- addendum_temporal.md（结果）
- addendum_temporal_progress.log（进度日志）
"""
import datetime
import numpy as np
import pandas as pd

BASE = '/home/karl/repos/personal/stock_qt_nd'
TRADES = f'{BASE}/experiments/divergence_seed_trial_history/trades_seed.parquet'
CAL = f'{BASE}/stock_data/daily/000001.SH.parquet'
LOG = f'{BASE}/experiments/divergence_seed_trial_history/addendum_temporal_progress.log'
MD = f'{BASE}/experiments/divergence_seed_trial_history/addendum_temporal.md'

SEEDS = ['S1', 'S2', 'S3', 'S4', 'S5']


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def fnum(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    if float(x).is_integer():
        return str(int(x))
    return f'{x:.{nd}f}'


def fpct(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x * 100:.{nd}f}%'


def pctile(vals, qs):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return [np.nan] * len(qs)
    return list(np.percentile(vals, qs))


log('START addendum_temporal（信号时间分布深挖）')

# ---------------------------------------------------------------- 数据装载
trades = pd.read_parquet(TRADES)
frame = trades[(trades.variant == 'v1') & (trades.H == 20)].copy().reset_index(drop=True)
assert len(frame) == 96577, len(frame)

cal_raw = pd.read_parquet(CAL)
cal = pd.to_datetime(cal_raw.trade_date, format='%Y%m%d').sort_values().reset_index(drop=True)
close = cal_raw.set_index(pd.to_datetime(cal_raw.trade_date, format='%Y%m%d')).loc[cal, 'close'].to_numpy()
NCAL = len(cal)
cal_pos = {d: i for i, d in enumerate(cal)}
assert NCAL == 8000, NCAL

frame['pos'] = frame.event_date.map(cal_pos)  # NaN = 早于日历起点
frame['in_cal'] = frame.pos.notna()

masks = {s: frame[f'sel_{s}'].to_numpy() for s in SEEDS}
masks['并集'] = np.logical_or.reduce([masks[s] for s in SEEDS])
CONFIGS = SEEDS + ['并集']

n_pre = int((~frame.in_cal).sum())
n_pre_sel = {c: int((masks[c] & ~frame.in_cal.to_numpy()).sum()) for c in CONFIGS}

# ---------------------------------------------------------------- 口径披露数据
disclosures = {
    'n_pre': n_pre,
    'n_pre_sel': n_pre_sel,
    'union_eq_s4': bool((masks['并集'] == masks['S4']).all()),
    'sel_all_alltrue': bool(frame.sel_ALL.all()),
}

MD_PARTS = []


def md(s=''):
    MD_PARTS.append(s)


# ================================================================ 维度 1：逐日信号数全分布
dim1_rows = []
dim1_conservation = {}
for c in CONFIGS:
    sel = frame[masks[c]]
    daily = sel.groupby('event_date').size()
    in_cal_days = int(sum(1 for d in daily.index if d in cal_pos))
    p25, p50, p75, p90, p95, p99 = pctile(daily.to_numpy(), [25, 50, 75, 90, 95, 99])
    dim1_rows.append([
        c, len(sel), len(daily), in_cal_days, fpct((NCAL - in_cal_days) / NCAL),
        fnum(p25), fnum(p50), fnum(p75), fnum(p90), fnum(p95), fnum(p99), int(daily.max()),
    ])
    dim1_conservation[c] = (int(daily.sum()), len(sel))
log('维度1 完成：逐日信号数全分布')

# ================================================================ 维度 2：爆发段结构
def find_segments(mask):
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return starts, ends


dim2_rows = []
seg_store = {}
for c in CONFIGS:
    sel = frame[masks[c] & frame.in_cal.to_numpy()]
    daily = sel.groupby('event_date').size()
    counts = np.zeros(NCAL, dtype=np.int64)
    for d, v in daily.items():
        counts[cal_pos[d]] = v
    starts, ends = find_segments(counts >= 1)
    lens = ends - starts + 1
    totals = np.array([counts[s:e + 1].sum() for s, e in zip(starts, ends)])
    gaps = starts[1:] - ends[:-1] - 1
    seg_store[c] = dict(starts=starts, ends=ends, lens=lens, totals=totals, gaps=gaps,
                        mask=counts >= 1)
    lp = pctile(lens, [25, 50, 75, 95])
    tp = pctile(totals, [50])
    gp = pctile(gaps, [25, 50, 75, 95]) if len(gaps) else [np.nan] * 4
    dim2_rows.append([
        c, len(starts),
        fnum(lp[0]), fnum(lp[1]), fnum(lp[2]), fnum(lp[3]), int(lens.max()),
        fnum(tp[0]), int(totals.max()),
        fnum(gp[0]), fnum(gp[1]), fnum(gp[2]), fnum(gp[3]),
        int(gaps.max()) if len(gaps) else '—',
    ])
log('维度2 完成：爆发段结构')

# 并集 top5 段 / top5 闲置期
u = seg_store['并集']
order_len = np.argsort(-u['lens'])[:5]
top5_seg = [[i + 1, cal[u['starts'][k]].strftime('%Y-%m-%d'), cal[u['ends'][k]].strftime('%Y-%m-%d'),
             int(u['lens'][k]), int(u['totals'][k])] for i, k in enumerate(order_len)]
order_gap = np.argsort(-u['gaps'])[:5]
top5_gap = [[i + 1, cal[u['ends'][k]].strftime('%Y-%m-%d'), cal[u['starts'][k + 1]].strftime('%Y-%m-%d'),
             int(u['gaps'][k])] for i, k in enumerate(order_gap)]
# 各配置最长段一览
longest_rows = []
for c in CONFIGS:
    s = seg_store[c]
    k = int(np.argmax(s['lens']))
    longest_rows.append([c, cal[s['starts'][k]].strftime('%Y-%m-%d'),
                         cal[s['ends'][k]].strftime('%Y-%m-%d'), int(s['lens'][k]),
                         int(s['totals'][k])])

# ================================================================ 维度 3：并发在持曲线
closed = frame[(frame.status == 'closed') & frame.in_cal].copy()
n_closed_drop = int(((frame.status == 'closed') & ~frame.in_cal).sum())
eidx = closed.entry_date.map(cal_pos).to_numpy()
xidx = closed.exit_date.map(cal_pos).to_numpy()
assert not np.isnan(eidx).any() and not np.isnan(xidx).any()

dim3_rows = []
curve_store = {}
closed_masks = {c: masks[c][closed.index.to_numpy()] for c in CONFIGS}
for c in CONFIGS:
    m = closed_masks[c]
    diff = np.zeros(NCAL + 1, dtype=np.int64)
    np.add.at(diff, eidx[m], 1)
    np.add.at(diff, xidx[m] + 1, -1)
    curve = np.cumsum(diff)[:NCAL]
    curve_store[c] = curve
    p50, p90, p95, p99 = pctile(curve, [50, 90, 95, 99])
    peak = int(curve.max())
    peak_date = cal[int(np.argmax(curve))].strftime('%Y-%m-%d')
    dim3_rows.append([c, int(m.sum()), fnum(p50), fnum(p90), fnum(p95), fnum(p99), peak,
                      peak_date, fpct(float((curve == 0).mean()))])
ucurve = curve_store['并集']
top10_idx = np.argsort(-ucurve)[:10]
top10_peak = [[i + 1, cal[k].strftime('%Y-%m-%d'), int(ucurve[k])] for i, k in enumerate(top10_idx)]
log('维度3 完成：并发在持曲线')

# 自检 3：抽查 3 个日期手工重算
clu = closed[closed_masks['并集']]
rng = np.random.default_rng(20260903)
spot_dates = [cal[int(np.argmax(ucurve))], cal[NCAL // 2], cal[int(rng.integers(0, NCAL))]]
spot_rows = []
spot_ok = True
for d in spot_dates:
    manual = int(((clu.entry_date <= d) & (clu.exit_date >= d)).sum())
    cv = int(ucurve[cal_pos[d]])
    ok = manual == cv
    spot_ok &= ok
    spot_rows.append([d.strftime('%Y-%m-%d'), manual, cv, 'PASS' if ok else 'FAIL'])

# ================================================================ 维度 4：单股重复触发
dim4_rows = []
for c in CONFIGS:
    sel = frame[masks[c] & frame.in_cal.to_numpy()][['ts_code', 'event_date', 'pos']]
    sel = sel.sort_values(['ts_code', 'event_date'])
    intervals = sel.groupby('ts_code')['pos'].diff().dropna().to_numpy()
    n_stock = sel.ts_code.nunique()
    p25, p50, p75 = pctile(intervals, [25, 50, 75]) if len(intervals) else [np.nan] * 3
    share20 = float((intervals <= 20).mean()) if len(intervals) else np.nan
    # 同股真实持仓重叠（closed 口径）：entry_pos <= 此前同股最大 exit_pos
    cl_c = closed[closed_masks[c]][['ts_code']].copy()
    cl_c['entry_p'] = closed.loc[cl_c.index, 'entry_date'].map(cal_pos)
    cl_c['exit_p'] = closed.loc[cl_c.index, 'exit_date'].map(cal_pos)
    cl_c = cl_c.sort_values(['ts_code', 'entry_p'])
    cl_c['prev_max_exit'] = cl_c.groupby('ts_code')['exit_p'].cummax().groupby(cl_c.ts_code).shift(1)
    overlap = cl_c.prev_max_exit.notna() & (cl_c.entry_p <= cl_c.prev_max_exit)
    dim4_rows.append([c, n_stock, len(intervals), fnum(p25), fnum(p50), fnum(p75),
                      fpct(share20), int(overlap.sum()), fpct(float(overlap.mean()) if len(cl_c) else np.nan)])
log('维度4 完成：单股重复触发')

# ================================================================ 维度 5：爆发段与市场背景
seg_u = seg_store['并集']
ret20_list, ret60_list = [], []
for s in seg_u['starts']:
    ret20_list.append(close[s - 1] / close[s - 21] - 1 if s >= 21 else np.nan)
    ret60_list.append(close[s - 1] / close[s - 61] - 1 if s >= 61 else np.nan)
ret20_arr = np.array(ret20_list)
ret60_arr = np.array(ret60_list)
r20p = pctile(ret20_arr, [25, 50, 75])
r60p = pctile(ret60_arr, [25, 50, 75])
n_r20 = int((~np.isnan(ret20_arr)).sum())
n_r60 = int((~np.isnan(ret60_arr)).sum())
share_le5 = float(np.nanmean(ret20_arr <= -0.05))
share_le10 = float(np.nanmean(ret20_arr <= -0.10))
# 对照：所有“前20日指数跌≥10%”的交易日中落在并集爆发段内的数量
day_ret20 = np.full(NCAL, np.nan)
for i in range(21, NCAL):
    day_ret20[i] = close[i - 1] / close[i - 21] - 1
down_days = day_ret20 <= -0.10
n_down = int(down_days.sum())
n_down_in_seg = int((down_days & seg_u['mask']).sum())
log('维度5 完成：爆发段与市场背景')

# ================================================================ 维度 6：逐年热力
years = list(range(1992, 2027))
frame['year'] = frame.event_date.dt.year
frame['ym'] = frame.event_date.dt.strftime('%Y-%m')
cal_years = cal.dt.year.to_numpy()
dim6_rows = []
for y in years:
    fy = frame[frame.year == y]
    row = [y]
    row.append(int(masks['并集'][frame.year.to_numpy() == y].sum()))
    for s in SEEDS:
        row.append(int(masks[s][frame.year.to_numpy() == y].sum()))
    n_seg = int(((cal[seg_u['starts']].dt.year == y)).sum()) if len(seg_u['starts']) else 0
    n_month = fy.loc[masks['并集'][frame.year.to_numpy() == y], 'ym'].nunique()
    cmask = cal_years == y
    peak = int(ucurve[cmask].max()) if cmask.any() else None
    row += [n_seg, n_month, fnum(peak) if peak is not None else '—']
    dim6_rows.append(row)
log('维度6 完成：逐年热力')

# ================================================================ 自检
exp = {'S1': 5580, 'S2': 2419, 'S3': 1159, 'S4': 11888, 'S5': 3271}
check1 = {s: int(masks[s].sum()) for s in SEEDS}
check1_ok = all(check1[s] == exp[s] for s in SEEDS) and len(frame) == 96577 and int(frame.sel_ALL.sum()) == 96577

f = frame
nested_pairs = [('S3', 'S2'), ('S2', 'S1'), ('S1', 'S4'), ('S2', 'S4'), ('S3', 'S5')]
nested = {f'{a}⊆{b}': int((f[f'sel_{a}'] & ~f[f'sel_{b}']).sum()) for a, b in nested_pairs}
check2_ok = all(v == 0 for v in nested.values())

check3_ok = spot_ok
check4 = {c: dim1_conservation[c] for c in CONFIGS}
check4_ok = all(a == b for a, b in check4.values())
log('自检完成')

# ================================================================ 写 addendum_temporal.md
md('# 背离v6 种子实验 —— 信号时间分布深挖（addendum_temporal）')
md()
md(f'生成时间：{datetime.datetime.now().isoformat(timespec="seconds")}。')
md('本附录为纯描述性统计：零拟合、零筛选决策，不对任何策略优劣作推论。')
md()
md('## 口径')
md()
md('- 事件级分析框：`trades_seed.parquet` 中 `variant==\'v1\' 且 H==20` 的子集，共 96577 行，每个 v1 事件一行。')
md('- 种子口径：S1~S5 分别取布尔列 `sel_S1`~`sel_S5`；并集 = 五列布尔或（H20 框一行一事件，天然去重）。')
md('- `sel_ALL` 列在 v1×H20 框内全为 True（=全事件池标记，96577），并非五种子并集；本附录"并集"均指五列布尔或。')
if disclosures['union_eq_s4']:
    md('- 实测并集在数值上等于 S4（11888 事件），因为嵌套关系 S3⊆S2⊆S1⊆S4 且 S5⊆S4、S3⊆S5 全部成立（见末尾自检 2）；并集行的数字因此与 S4 行一致，仍按要求单独列示。')
md('- 市场日历：`stock_data/daily/000001.SH.parquet`（上证指数日线），1993-10-08 ~ 2026-08-31，共 8000 个交易日；指数收益用该文件 close 计算。')
md(f'- 日历起点之前有 {n_pre} 个事件（最早 1992-12-25，均为 1992-12 ~ 1993-09 的上交所早期事件），其中被种子选中的数量：'
   + '、'.join(f'{c}={n_pre_sel[c]}' for c in CONFIGS)
   + '。凡需对齐市场日历的分析（维度 2 段结构、维度 3 并发、维度 5 市场背景）均剔除这些无法落日历的事件；维度 1/4/6 按 event_date 自然口径保留并在表内单列披露。')
md('- 爆发段定义：市场日历上连续交易日每日 ≥1 信号（按所涉口径计）的最长区间；段间距 = 上一段末日到下一段首日之间的零信号交易日数（即资金闲置期）。')
md('- 并发在持口径：仅 `status==\'closed\'` 的交易；持仓区间 = 市场日历上 [entry_date, exit_date] 闭区间内的全部交易日（个股日线无停牌行，此口径不含停牌占用；与主报告口径一致）。'
   f'closed 但无法落日历的交易 {n_closed_drop} 笔（即日历起点前的事件）已剔除。')
md('- 并集口径下同一股票两笔重叠持仓按两笔计（对应真实决策中的同股加仓），重叠规模在维度 4 单独量化。')
md()

md('## 维度 1：逐日信号数全分布')
md()
md('按 event_date 聚合；零信号日占比的分母 = 市场日历全部 8000 个交易日；"落在日历内的有信号日数"为该占比的分子。')
md()
md('| 配置 | 事件总数 | 有信号日数 | 其中落在日历内 | 零信号日占比 | 日信号数P25 | P50 | P75 | P90 | P95 | P99 | max |')
md('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in dim1_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()

md('## 维度 2：爆发段结构')
md()
md('段长与段间距单位均为交易日；段间距一行统计的是相邻两段之间的闲置期（段数−1 个样本）。')
md()
md('| 配置 | 段数 | 段长P25 | P50 | P75 | P95 | max | 段内信号P50 | 段内信号max | 段间距P25 | P50 | P75 | P95 | max |')
md('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in dim2_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()
md('并集口径最长的 5 个爆发段：')
md()
md('| 排名 | 段首日 | 段末日 | 段长(交易日) | 段内信号数 |')
md('|---|---|---|---|---|')
for r in top5_seg:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()
md('并集口径最久的 5 个闲置期（段间距）：')
md()
md('| 排名 | 上一段末日 | 下一段首日 | 闲置交易日数 |')
md('|---|---|---|---|')
for r in top5_gap:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()
md('各配置最长段一览：')
md()
md('| 配置 | 最长段首日 | 最长段末日 | 段长(交易日) | 段内信号数 |')
md('|---|---|---|---|---|')
for r in longest_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()

md('## 维度 3：并发在持曲线（决定需要几个仓）')
md()
md('分母 = 市场日历全部 8000 个交易日（含在持为 0 的日子）；曲线逐日值 = 当日在持笔数（同一股票两笔重叠持仓计两笔）。')
md()
md('| 配置 | closed笔数 | 在持P50 | P90 | P95 | P99 | max | 峰值日期 | 在持为0的日历日占比 |')
md('|---|---|---|---|---|---|---|---|---|')
for r in dim3_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()
md('并集口径在持峰值 top10 日期：')
md()
md('| 排名 | 日期 | 在持笔数 |')
md('|---|---|---|')
for r in top10_peak:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()

md('## 维度 4：单股重复触发')
md()
md('间隔 = 同一 ts_code 相邻两次被选中事件之间的交易日数（按市场日历计）；间隔 ≤20 交易日即与上一笔 H20 持仓存在潜在重叠。')
md('末两列为同股真实持仓重叠的精确量化（closed 口径：本笔 entry 不晚于同股此前持仓的最大 exit）。')
md()
md('| 配置 | 涉及股票数 | 间隔样本数 | 间隔P25 | P50 | P75 | 间隔≤20交易日占比 | 同股真实重叠笔数 | 占closed笔数比 |')
md('|---|---|---|---|---|---|---|---|---|')
for r in dim4_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()

md('## 维度 5：爆发段与市场背景')
md()
md('对象 = 并集口径全部爆发段；段首日前 N 交易日指数收益 = close[段首日前1日] / close[段首日前N+1日] − 1（上证指数）。')
md(f'段总数 {len(seg_u["starts"])}；段首日前 20 日收益可算 {n_r20} 段、前 60 日可算 {n_r60} 段（段首距日历起点不足的段计 NaN 并剔除）。')
md()
md('| 指标 | P25 | 中位数 | P75 |')
md('|---|---|---|---|')
md(f'| 段首日前20日指数收益 | {fpct(r20p[0])} | {fpct(r20p[1])} | {fpct(r20p[2])} |')
md(f'| 段首日前60日指数收益 | {fpct(r60p[0])} | {fpct(r60p[1])} | {fpct(r60p[2])} |')
md()
md(f'- 段首日前 20 日指数收益 ≤ −5% 的段占比：{fpct(share_le5)}（{int((ret20_arr <= -0.05).sum())}/{n_r20}）。')
md(f'- 段首日前 20 日指数收益 ≤ −10% 的段占比：{fpct(share_le10)}（{int((ret20_arr <= -0.10).sum())}/{n_r20}）。')
md(f'- 对照：市场日历上"指数前 20 日跌 ≥10%"的交易日共 {n_down} 天，其中落在并集爆发段内 {n_down_in_seg} 天（{fpct(n_down_in_seg / n_down if n_down else np.nan)}）。')
md()

md('## 维度 6：逐年热力（1992~2026）')
md()
md('段数按段首日所在年份计；并发在持峰值取自并集口径逐日在持曲线（1992 年无日历覆盖，1993 年仅覆盖 10-08 起）。')
md()
md('| 年份 | 并集信号数 | S1 | S2 | S3 | S4 | S5 | 段数 | 有信号月数 | 并发在持峰值 |')
md('|---|---|---|---|---|---|---|---|---|---|')
for r in dim6_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()

md('## 自检')
md()
md(f'1. 事件总数对账：' + '、'.join(f'{s}={check1[s]}（期望{exp[s]}）' for s in SEEDS)
   + f'；v1×H20 框总行数={len(frame)}、`sel_ALL` 求和={int(frame.sel_ALL.sum())}（期望 96577）。'
   + ('PASS' if check1_ok else 'FAIL'))
md(f'2. 嵌套关系：' + '、'.join(f'{k} 违反数={v}' for k, v in nested.items())
   + '（另测得 S5⊆S4 违反数=' + str(int((f.sel_S5 & ~f.sel_S4).sum())) + '，故并集=S4）。'
   + ('PASS' if check2_ok else 'FAIL'))
md('3. 并发在持曲线抽查（并集口径，从交易明细按 entry_date ≤ d ≤ exit_date 手工重算对比曲线值）：')
md()
md('| 日期 | 手工重算在持笔数 | 曲线值 | 结果 |')
md('|---|---|---|---|')
for r in spot_rows:
    md('| ' + ' | '.join(str(x) for x in r) + ' |')
md()
md(f'4. 守恒：维度 1 日信号数合计 vs 事件总数 —— ' + '、'.join(f'{c}: {a}={b}' for c, (a, b) in check4.items())
   + ('。PASS' if check4_ok else '。FAIL'))
md()
md('**ALL CHECKS ' + ('PASS' if all([check1_ok, check2_ok, check3_ok, check4_ok]) else 'FAIL') + '**')

with open(MD, 'w') as fh:
    fh.write('\n'.join(MD_PARTS) + '\n')

log('addendum_temporal.md 写出完成')
log('ALL DONE')
