# v6 战役终审 · 预登记规格(冻结)

冻结时间:2026-09-11,先于任何 test 段结果产出。
本文件与 `run_final_exam.py` 在同一 commit 落盘,commit 哈希即预登记时间戳。
图:issue #55;本票:#56。

## 1. 终审问题

「模型A × 种子并集 × H20」在零触碰 test 段是否过 #31 五线且胜对照 +2pp。
单发:无重试、无修指标、无换格子。任一不过 = CAMPAIGN_DEAD。

## 2. 数据与口径(逐字锚定存档)

- 交易:`experiments/divergence_seed_trial_history/trades_seed.parquet`,
  过滤 `variant=='v1' & H==20 & status=='closed'`(= #31 存档口径,v1 引擎/净扣成本;
  全历史 95,386 笔,(ts_code,event_date) 键零重复,已核)。
- 模型A 分数:`experiments/v2_on_v6_rerun/cache/scores_v2on6_p7fix.parquet`
  (md5 b1f06afac49a32ad97e0e4f89f18ecd5;v2-stack 修复后;train=OOF、val 未参训)。
- 模型B 分数:`experiments/v6_model_campaign/m4_training_selection/scores_v6.parquet`(M线 10 特征)。
- 终审段:`seg=='test'`(2024+;池事件 17,902;A/B 分数零 NaN,已核)。
- 连接:trades ⋈ scores on (ts_code, event_date==date)。

## 3. 事件集与切片(配置为行)

排名规则(全表统一,锚定监工 D3 注册口径):排名总体内按 score 降序、ts_code 升序、event_date 升序,mergesort 稳定排序,取 `head(ceil(n·x))`。

| 行 | 切片 | 角色 |
|---|---|---|
| pool_ALL | test 段池内全部成交 | 信息(基线) |
| union_S4 | 种子并集(sel_S4,= v6-4 超集)全取 | **对照** |
| seed_S1~S5 | 各种子 sel_S1~sel_S5 | 信息 |
| AxS4_top30 | 并集内模型A top30% | **唯一宣判格(主格子)** |
| AxS4_top10 / AxS4_top50 | 并集内模型A top10% / top50% | 信息 |
| Axpool_top5 / Axpool_top10 | 池内模型A top5% / top10% | 信息 |
| BxS4_top10 / BxS4_top30 / BxS4_top50 | 并集内模型B 三档 | 信息 |

主格子选 top30% 的依据(冻结前计数,仅计数未碰结果列):test 并集成交 2,103 笔,top30%≈631 笔(n≥300 余量充足);top10%≈211 笔,按 c1 线天然不达,只做信息行。

## 4. 宣判线(逐字 #31 存档,run_seeds.py L668-697)

- c1:n_closed ≥ 300
- c2:net_mean > 0
- c3:cluster_t ≥ 2(Liang-Zeger,按 entry_date 聚类,逐字 run_seeds.py L437-451)
- c4:win_year_share ≥ 0.6(按 entry_date 年分组,年成交 ≥30 笔才计资格,资格年内净笔均 >0 的占比;run_seeds.py L555-569)
- c5:date_conc ≤ 0.5(top5 入场日 net_pnl 合计 / 全部 net_pnl 合计,合计 ≤0 时记 NaN 即不过;run_seeds.py L570-577)

## 5. 宣判规则

- 主格子 AxS4_top30:c1~c5 全过 **且** net_mean ≥ 对照 union_S4 的 net_mean + 0.02 → `ROUTE_ALIVE`。
- 任一不过 → `CAMPAIGN_DEAD`(单发,无重试)。
- 信息行不承载宣判。

## 6. 研究自由度披露

- 格子(模型×种子并集×H20)与档位(top30%)由 val 段证据选出
  (val:A×种子 top30% 胜率 61.89%/cluster_t +2.64、净笔均 +4.69% vs 并集 +1.88%;top10% 65.82%/t+3.21 但 n=158)。
- test 段此前仅看过事件计数(本文件 §3 功效设计),结果列零触碰。
- 既往判死(策略层 #32~#40、M1~M5 #52、v2 复刻 #54)均非本终审格,不构成本格的预断;本格从未终测。
- 已知限制:test 段跨 3 个日历年(2024/2025/2026,2026 为部分年,数据至 2026-08),c4 资格年粒度粗;c5 在 val 段曾对一切切片(含种子自身)失败,test 段是何表现未知——这正是终审要回答的。

## 7. 产出

- `results_final_exam.csv`(配置为行全表)
- `verdict_final_exam.json`(五线逐条 + 边际 + 宣判 + 输入文件 md5 台账)
- 落盘 commit 后,监工独立复核(#58,自写重算,不信任本脚本)通过方可宣判(#59)。
