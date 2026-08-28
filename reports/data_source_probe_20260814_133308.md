# tinyshare 数据源权限探测报告

- 探测时间：2026-08-14 13:33:06
- Token（末 6 位）：`…7ade48`
- 汇总：✅ 10 可用 / ❌ 0 不可用 / ⚠️ 0 返回空
- go/no-go 判定：**🟢 GO**

---

## 接口探测明细

| 状态 | 接口 | 行×列 | 首行样本 / 错误 |
| --- | --- | --- | --- |
| ✅ OK | `stock_basic(list_status=L)` | 5543×4 | ts_code=000001.SZ  name=平安银行  list_date=19910403  industry=银行 |
| ✅ OK | `stock_basic(list_status=D)` | 341×4 | ts_code=000003.SZ  name=PT金田A(退)  list_date=19910703  delist_date=20020614 |
| ✅ OK | `daily(000001.SZ, 20260804~20260814)` | 8×11 | ts_code=000001.SZ  trade_date=20260813  open=11.23  high=11.27  low=11.18  close=11.25 |
| ✅ OK | `daily_basic(000001.SZ, 20260804~20260814)` | 8×6 | ts_code=000001.SZ  trade_date=20260813  pe=5.1208  pb=0.4704  total_mv=21831657.975  turnover_rate=0.3896 |
| ✅ OK | `moneyflow(000001.SZ, 20260804~20260814)` | 8×20 | ts_code=000001.SZ  trade_date=20260813  buy_sm_vol=210829  buy_sm_amount=23662.04  sell_sm_vol=255558  sell_sm_amount=28 |
| ✅ OK | `moneyflow_hsgt(20260804~20260814)` | 8×7 | trade_date=20260813  ggt_ss=31801.1  ggt_sz=22978.06  hgt=144941.15  sgt=169832.36  north_money=314773.51 |
| ✅ OK | `stk_factor(000001.SZ, 20260804~20260814)` | 8×35 | ts_code=000001.SZ  trade_date=20260813  close=11.25  open=11.23  high=11.27  low=11.18 |
| ✅ OK | `daily(退市股 000003.SZ, 20010614~20020613)` | 17×11 | ts_code=000003.SZ  trade_date=20020426  open=2.71  high=2.71  low=2.71  close=2.71 |
| ✅ OK | `daily(退市股 000013.SZ, 20030921~20040919)` | 143×11 | ts_code=000013.SZ  trade_date=20040428  open=2.45  high=2.55  low=2.38  close=2.47 |
| ✅ OK | `daily(退市股 000015.SZ, 20001022~20011021)` | 30×11 | ts_code=000015.SZ  trade_date=20010608  open=6.85  high=6.85  low=6.85  close=6.85 |

---

## 退市股历史日线可得性

抽样股票：`000003.SZ`, `000013.SZ`, `000015.SZ`

- ✅ `daily(退市股 000003.SZ, 20010614~20020613)`：17 行
- ✅ `daily(退市股 000013.SZ, 20030921~20040919)`：143 行
- ✅ `daily(退市股 000015.SZ, 20001022~20011021)`：30 行

---

## 结论与建议

所有接口均可用，含退市股历史日线。可进入数据层重建（issue #5）。
