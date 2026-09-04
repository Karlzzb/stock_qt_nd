# 池内排序探针（ranking probe）——预登记

登记时间：2026-09-04，先于任何跑数落盘。
本实验为判决性实验：回答"背离事件池内是否存在可用单因子排序出的正期望尾部"。
阴性结果（没有任何因子过线）是合法且有价值的交付，严禁为了出阳性而加码挖掘。

## 数据口径

- 事件级逐笔盈亏：`experiments/divergence_anchor_eval_2026/backtest/event_study.parquet`，仅用 `status=="closed"` 行。
  每行 = 一事件一出场规则：`signal`(events_v1/events_v2) / `config`(A13/B15/E1-H12) / `ts_code` / `event_date` / `ret`（净收益，含全部成本）/ `excess`（相对 000905.SH 同窗）等。
- 事件表：`events_v1.parquet` / `events_v2.parquet`，含 cross_prev_date, cross_prev_dif, cross_date, cross_dif, dif_lift, anchor_date, anchor_close（v2 另有 anchor_prev_date/anchor_prev_close）。
- 行情：`stock_data/daily/*.parquet` 未复权日线，schema 过滤（需同时有 ts_code 与 vol 列；<100 行跳过），按 ts_code 与事件表 join。
- 全部因子仅使用 ≤ event_date 的行情与事件字段计算，因果。
  特征计算函数内置硬断言：用于特征的行情最大日期 == event_date（防泄漏）。

## 目标变量

- 主目标 = E1-H12（裸持 12 日）净收益 ret。
- 副目标 = E1-H12 的 excess（只出描述统计，不作判活依据）。
- 稳健性对照 = A13 的 ret（只对主目标上显著的因子复查方向一致性）。

## 因子清单（12 个，v2 追加 f13）

- f1 dif_lift（已知有害，作对照）
- f2 cross_dif（金叉处 DIF 水位）
- f3 cross_dif / close[event]（价格归一化水位）
- f4 金叉间隔交易日数（cross_date − cross_prev_date，按该股自身交易日位置差）
- f5 事件日距锚点交易日数（event_date − anchor_date，按该股自身交易日位置差）
- f6 锚点反弹幅度 close[event]/anchor_close − 1（已吃掉的肉，核心嫌疑因子）
- f7 锚点 52 周位置 (anchor_close − min250)/(max250 − min250)，min/max 取截至 event_date（含）的 250 个交易日 close；历史不足 250 日记 NaN
- f8 入锚回撤 anchor_close/max(close) − 1，max 取 [event_date 前 60 交易日 .. anchor_date] 区间 close
- f9 事件前 20 日动量 close[event]/close[event−20] − 1
- f10 事件日量比 vol[event]/mean(vol[event−20..event−1])
- f11 20 日波动率 std(close 日收益[event−19..event])/日（由 close 计算，ddof=1）
- f12 事件日成交额 amount[event]
- f13（仅 v2）锚对锚跌幅 anchor_close/anchor_prev_close − 1

## 检验方法

每变体（v1、v2）× 每因子（v1: 12 个，v2: 13 个），在主目标（E1-H12 ret，closed）上：

1. 按因子值十分位分组（等频，pd.qcut，D1=最低档），出各档 ret 均值/中位/胜率/excess 均值。
2. Spearman 秩相关 ρ（因子值 vs ret）及 t 值，t = ρ·sqrt((n−2)/(1−ρ²))。
3. 头部切片统计：top 十分位 + top 五分位（按因子值高端）的 ret 均值、excess 均值、单样本 t 值、笔数；
   同时落盘 bottom 五分位同样统计，保证双向透明。
4. 多重检验门槛：12~13 因子 Bonferroni，显著性要求 p < 0.004（|t| ≳ 2.9）。

## 判活/判死标准（预登记，不得事后移动）

某因子"有效"需同时满足以下三条（均在主目标 E1-H12 ret 上）：

1. |Spearman t| ≥ 2.9 且方向单调可解释（十分位均值对档位序号的 Spearman 相关符号与 ρ 一致）；
2. 方向有利端五分位 ret 均值 > 0 且该切片单样本 t ≥ 3；
3. 方向有利端五分位 ret 均值 − 池均值 ≥ +1pp（经济意义门槛，池均值 = 该变体 E1-H12 closed 全样本 ret 均值）。

预登记澄清（登记时钉死，非事后移动）："方向有利端"由 Spearman ρ 符号决定——ρ>0 时取因子值 top 五分位，ρ<0 时取 bottom 五分位。
原因是本探针的目的是"挑出正期望单子"，一个负向因子若能反向挑出正期望尾部同样是可排序结构；
两个方向的切片统计均原样落盘，不隐藏任何一端。

- 无任何因子过线 → 判死（2026 沙盒内无可排序结构）。
- 过线因子 → 用 A13 的 ret 复查方向一致性（Spearman ρ 符号相同），仍只作"探索期线索"，注明必须在战役验证段重验。

## 自检（管线正确性闸门，先于判读）

1. 精确复现：用本管线 f1（dif_lift）在该变体 E1-H12 closed 样本上按三分位分组，
   复算 ret 均值/中位/胜率/excess 均值，必须与既有产物 `backtest/event_study_terciles.csv` 对应行逐一一致（容差 1e-6）。
2. 方向一致：十分位层面，v1 高档（D8–D10 合并）ret 均值不优于低档（D1–D3 合并）超过 +0.5pp，
   且 v2 的 f1 最高档（D10）ret 均值为全部十档中最差或次差。
   不一致即停并报告，说明特征管线有 bug，不进入判读。

## 产物

- `features.parquet`（每事件一行：signal/ts_code/event_date/f1..f13）
- `decile_tables.csv`（signal × factor × decile：n, ret 均值/中位/胜率, excess 均值）
- `spearman.csv`（signal × factor：n, rho, t）
- `top_slice.csv`（signal × factor × slice：top_decile / top_quintile / bottom_quintile 的 n, ret 均值, excess 均值, t）
- `verdict.json`（每因子三条判活标准逐项结果 + 过线与否 + 总判定 + 自检结果）
