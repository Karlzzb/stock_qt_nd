# strategy_engine_v2 校验报告（issue #19 择时覆盖层冻结验收）

校验时间：2026-09-02 11:50:49，总耗时 29.5 秒。
总体结论：全部通过，引擎 v2 冻结。

- [PASS] regression_timing_none_vs_18_archive：B1(backup_S3_E1_N10_H20): equity[rows=928 max_abs_diff=0.000e+00] trades[rows=304 max_abs_diff=0.000e+00] final_equity=1949498.835437 (归档 1949498.835437)；B2(backup_S1_E1_N10_H20): equity[rows=928 max_abs_diff=0.000e+00] trades[rows=352 max_abs_diff=0.000e+00] final_equity=1748327.122768 (归档 1748327.122768)
- [PASS] timing_synthetic_tests：(a-G1) 降档后 0 笔新入场，5 仓全部 E1@d7 自然退出；(a-G2) 4 仓 d5 强制退出(G2_forced)，T05 跌停顺延 d6，forced_exits=5 forced_deferred=1；(b) round_slots(10,0.5)=5；T3 半敞口下 8 信号只入 sig_idx 前 5；(c) 仅 100 日历史时 T1 全程降级全敞口，fallback=10 天，正常入场 2 仓；(d) 5->3 槽时浮盈最低 2 仓(T01,T02) d5 强退，其余保留至 E1
- [PASS] leakage_exposure_replay：四规则敞口与 pandas shift/rolling 独立重算逐日一致，max_src_date 全部 < 决策日；T1 分布{0.0: 299, 1.0: 629}；T2 分布{0.0: 431, 1.0: 497}；T3 分布{0.0: 100, 0.5: 350, 1.0: 478}；T4 分布{0.5: 110, 1.0: 818}
- [PASS] smoke_B1_T1_G1_val：敞口分布{0.0: 299, 1.0: 629}，强制退出 0 笔（G1 应为 0），闸门拦截 132 笔，入场 251，曲线 1000000 -> 1269683，产物落 /home/karl/repos/personal/stock_qt_nd/v3_pipeline/reports/exposure_timing/smoke_B1_T1_G1

## 口径备注
- 择时信号源为中证500（000905.SH）收盘价，t 日敞口只用 trade_date <= t-1 的指数收盘，引擎内硬断言 max_src_date < t。
- T4 已实现波动 = 日收益（close.pct_change）20 日滚动样本 std（ddof=1）× sqrt(252)，分位窗口为含 t-1 当日的 250 个波动值的 90 分位。
- 窗口历史不足（T1<200、T2<60、T3<250、T4<270 个 <=t-1 收盘）降级为全敞口并记 warning（timing_fallback_days）。
- 槽位 = round_half_up(N × 敞口) = floor(x+0.5)，10×0.5=5、5×0.5=3。
- G1 只拦新入场（信号截取按入场日槽位、T+1 入场按当日槽位两处拦截），旧仓按原出场规则自然退出。
- G2 每日若持仓数 > 当日槽位，超出部分按收盘价强制退出（exit_reason=G2_forced），持有浮盈最低者优先（平局 ts_code 升序），遵守 T+1 不可卖与跌停顺延（次日重检）。
- 单仓预算恒为信号日 T 收盘权益 / N，不随敞口变化。
- timing=None 时 v2 直接委托 v1 run_config，落盘与 v1 逐字节一致。
- G2 强制退出不可能落在买入当日：降档日持仓数已达新槽位上限时新入场已被闸门拦截，T+1 分支为防御性保留。
