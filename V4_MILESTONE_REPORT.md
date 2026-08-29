# V4 里程碑报告：可信基线建立

**日期**：2026-08-29
**状态**：里程碑定稿。
**性质**：这不是策略成功的里程碑，而是**可信基线建立**的里程碑——本仓库历史上第一份无泄漏、可复现、口径诚实的模型评估结果。

---

## 1. 一句话结论

模型对 A 股横截面有**真实但微弱**的排名能力（30d Rank IC 0.07-0.08），毛超额收益无法覆盖全换手交易成本，**当前形态不可实盘**；主要矛盾是换手率成本，不是信号的有无。

## 2. 版本史：从假成功到真基线

| 版本 | 数据 | 结论 | 问题 |
|---|---|---|---|
| v3_0_1 | 原 cache（label 含 +15% 止盈截断） | Stage 1-4"完成"（#29-31） | 截断标签失真 |
| v3_0_3_no_cap | 删除 32.16% 截断样本的 cache | 宣称"READY FOR DEPLOYMENT"，年化数万% | **特征级标签泄漏（本文实锤作废）** |
| v3_0_4_truly_clean | 同上，堵漏重训 | IC 0.17-0.19 但 Long-only 亏损 | 评估 universe 有"删赢家"偏差 |
| **v4_0_0_clean** | 标签从原始日线重算，不删任何行 | 本报告 | 无已知数据问题 |

## 3. 泄漏实锤：v3_0_3_no_cap 为什么作废

训练的 631 个特征包含 `stop_loss_return_{3d..30d}`。
这些列由 `src/feature_pipeline_v2.py:415` `_calculate_stop_loss_return` 用**未来 N 天价格**模拟止损/止盈算出。
排除清单（`v3_pipeline/scripts/train_ranking.py` system_patterns + config exclude_patterns）只挡了 `^future_return_`/`^rank_future_return_`/`^label_`，漏掉了它们。

证据链（no-cap cache，951,303 行）：

| 证据 | 数值 |
|---|---|
| `stop_loss_return_3d` 与 label `future_return_3d` 完全相等的行占比 | **99.87%** |
| 两者相关系数 | 0.9991 |
| 模型 gain 占比：`stop_loss_return_3d` / `_5d` | **73.3% / 17.5%**（合计 >90%） |

反事实实验（同模型、同测试集、同评估路径，248 期）：

| 实验 | Rank IC | Top1 平均 3d 收益 | 胜率 |
|---|---|---|---|
| A. 原始特征（复现原报告） | 0.986 | +8.45% | 100% |
| B. 抹掉全部 7 列 stop_loss_return_* | **-0.116** | **-2.34%** | 34.3% |
| C. 只随机打乱 stop_loss_return_3d 一列 | 0.109 | +0.22% | 53.2% |

结论：模型的全部"预测能力"来自这几列未来数据。
之前 n 轮审查漏掉它的原因：`check_v3_leakage.py` 只按列名关键词（future/next/forward/target/label）扫描，`stop_loss_return` 不含这些词。

## 4. 同时被推翻的方法论错误

1. **伪 Sharpe**：原报告用"年化收益/年化波动"算出 Sharpe 4017；正确口径是 mean(期收益)/std(期收益)×√(每年期数)。
2. **重叠窗口复利**：248 个日度 timestamp 配 3d horizon 直接复利，自相关把胜率/MaxDD 统计 inflate 到失真；正确做法是相位切分不重叠窗口。
3. **删赢家偏差**：`v3_fix_remove_capped.py` 删除"任一 horizon 触及 +15%"的样本（32.16%），等于系统性删除赢家，universe 3d 均值从 -0.20% 恶化到 -1.82%。
4. **评估环境虚高**：v3_0_4 在删赢家 cache 上 IC 0.19，同一模型打到 V4 完整 universe 上 IC 降到 0.088——约一半"能力"来自更好猜的评估环境。

## 5. V4 干净链路

`v3_pipeline/scripts/build_v4_clean_cache.py`（原 `v4_rebuild_clean_labels.py`）：

1. 从 `stock_data/daily/` 5883 只股票的原始日线重算 7 个 horizon 的 future_return（纯 close-to-close：t 收盘买、t+1+h 收盘卖）。
2. 合并进原 cache 覆盖旧 label 列，匹配率 99.90%，**不删任何行**。
3. 物理删除全部 21 个未来派生列（`stop_loss_return_*`、`stop_loss_sell_date_*`、`future_sell_date_*`）。
4. 重算横截面 rank labels（`v3_pipeline/src/ranking_labels.py`）。
5. 产出 `v3_pipeline/feature_cache_v4_clean.parquet`（1,402,267 行 × 736 列，约 1.4G，不入库）。

重训配置：`v3_pipeline/configs/v4_0_0_clean.yaml`，exclude_patterns 永久包含 `^stop_loss_return_`（防御性保留，虽然列已物理删除）。
训练命令见配置文件头注释。
模型健康度：早停 64-81 轮，val NDCG@5 = 0.40-0.43（泄漏版为 0.999）；feature importance 无单列主导（top 特征 `distance_to_support` 仅占 6.3% gain）。

## 6. 最终诚实指标（v4_0_0_clean，624 特征）

评估口径：不重叠窗口（相位切分）、正规 Sharpe、交易成本 0.26%/期双边（全换手）、A 股不可做空个股。
复现命令：`python v3_pipeline/scripts/evaluate_honest.py v3_pipeline/models/v4_0_0_clean`
（本表所有数字为该脚本输出，脚本即口径的唯一事实来源。）

### 排名能力

| horizon | 验证集 Rank IC | 测试集 Rank IC | 验证集 ICIR | 测试集 ICIR | 测试集 IC>0 占比 |
|---|---|---|---|---|---|
| 3d | +0.015 | +0.026 | 0.09 | 0.16 | 57.1% |
| 10d | +0.045 | +0.073 | 0.28 | 0.49 | 70.4% |
| 30d | +0.084 | +0.075 | 0.55 | 0.48 | 69.1% |

IC 随 horizon 拉长而上升，10d/30d 的 IC>0 占比约 70%。

### 多空毛差与超额（每期）

| horizon | Top10−Bot10（验证） | Top10−Bot10（测试） | Top10−universe（验证） | Top10−universe（测试） |
|---|---|---|---|---|
| 3d | +1.03% | +1.37% | +0.31% | +0.36% |
| 10d | +1.38% | +2.04% | +0.27% | +0.43% |
| 30d | +4.62% | +0.31% | +2.25% | **−0.89%** |

### Long-only 净年化（含 0.26%/期成本，不重叠窗口）

| horizon | 组合 | 验证集 2022-01~2025-07 | 测试集 2025-08~2026-08 | 测试集基准年化 |
|---|---|---|---|---|
| 3d | Top1 | +8.0% | −14.7% | −6.2% |
| 3d | Top5 | +18.2% | −8.1% | −6.2% |
| 3d | Top10 | −4.1% | −2.2% | −6.2% |
| 10d | Top1 | −8.2% | +3.1% | −6.4% |
| 10d | Top5 | −0.0% | −0.1% | −6.4% |
| 10d | Top10 | −5.1% | −3.6% | −6.4% |
| 30d | Top1 | +8.9% | −57.7% | −7.0% |
| 30d | Top5 | +11.7% | −23.5% | −7.0% |
| 30d | Top10 | +12.9% | −16.6% | −7.0% |

### 判读

1. 信号是真的：全 horizon IC 为正、多空毛差为正、测试集不衰减（10d 还上升）。
2. 信号很弱：3d IC 仅 0.02-0.03，毛超额 +0.3%/期 vs 全换手成本 0.26%/期（≈22%/年），净收益在零附近。
3. **30d 测试集出现 IC 为正但 Top-N 尾部跑输 universe 的背离**（Top10−universe −0.89%/期）——IC 衡量整体单调性，不保证头部组合盈利；这再次印证"IC/ICIR 不能直接预测盈利"。
4. 换手率成本是当前主要矛盾，不是信号强度。

## 7. 教训清单（后续一切模型工作的准入纪律）

1. 特征审查按**数据血缘**（provenance）逐列回答"t 时刻可得吗"，禁止按列名关键词扫描。
2. 模型 feature importance 单列 >20% gain 即触发血缘审查。
3. 任何特征与 label 的 |corr| > 0.5 直接当泄漏处理。
4. 回测窗口必须与 horizon 不重叠（相位切分）；Sharpe 只能用 mean/std×√N。
5. 任何"删除样本"的数据清洗必须检查删除方向的收益分布（防删赢家偏差）。
6. 评估指标固定在排名口径（Rank IC/ICIR/多空差），但上线判据必须看不重叠窗口的**净收益**。
7. 禁用泄漏时代的任何产物作为对照组（包括"我们已经比它好了"式对比）。

## 8. 本次里程碑清理说明

- 删除：泄漏时代一次性剧本（`*_strategy_search.py`、`eval_*_no_cap.py`、`train_3d_no_cap.py`、`v3_fix_remove_capped.py` 等）、作废报告（`V3_FINAL_REPORT_NO_CAP.md`、`V3_FINAL_AUDIT_REPORT.md`、`V3_AUDIT_EXECUTIVE_SUMMARY.md`）、旧 V2 模型产物（`models/`）、泄漏模型目录（`v3_0_0`~`v3_0_3`）。
- 保留：`v3_pipeline/` 框架（scripts/src/backtest/tests）、`v3_pipeline/scripts/build_v4_clean_cache.py`、`evaluate_honest.py`、`v4_0_0_clean.yaml` 配置、`v3_pipeline/results/` 小 JSON（可追溯已关闭 issue 的历史记录）。
- 数据与模型二进制不再入库：`*.parquet`、`*.pkl`、`v3_pipeline/models/` 已加入 `.gitignore`，并用 git filter-repo 从全部历史中剥离（`.git` 由 7.3G 瘦身）。
- `v4_0_0_clean` 与 `v3_0_4_truly_clean` 模型文件保留在本机 `v3_pipeline/models/`（被 .gitignore 忽略，不提交）。

## 9. 下一步路线（新 issue 的任务池）

按优先级：

1. **降换手**：换仓带宽（排名变化超阈值才交易）、更大 Top-N、更长 horizon——目标是把 +0.3%/期 的毛超额从成本嘴里救出来。
2. **头部组合诊断**：30d 测试集 IC 正但 Top-N 负超额的背离，查头部选股的分布特征（是否集中于高波动/超跌反弹标的）。
3. **股指期货/ETF 对冲回测**：A 股个股不可做空，但可用指数工具对冲 beta，把横截面 alpha 剥离出来——这是排名信号最自然的变现方式。
4. **特征血缘审计落地**：把第 7 节的准入纪律写成脚本（每列标注 t 时刻可得性），并入 CI。
5. 数据重复审计：no-cap cache 时代发现 23.5% 重复 (timestamp, symbol)，V4 cache 需重新确认。
