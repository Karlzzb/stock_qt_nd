"""
评估协议：walk-forward 季度滚动重训 + 无菌终审段管控。

背景（PRD §评估协议）
---------------------
现有回测的评估协议有两大缺陷：
1. 模型只训练一次，无滚动重训——每个交易日的分数来自见过全部数据的模型。
2. 策略参数在测试+验证集上做网格搜索后挑最优回填——评估集已被过拟合。

本模块提供两个构件来修复上述问题：

WalkForwardProtocol
    季度粒度的滚动切分：每个季度末用截至该时点的全量数据重训，对下一季度打分。
    保证每个交易日的分数都来自「只见过该日之前数据」的模型。
    同时强制「无菌终审段」约束：终审段从不出现在任何训练/搜索窗口内。

EvaluationLedger
    持久化终审段评估台账。
    每个候选方案（特征集 / 参数组合 / 策略配置）在终审段上只允许评估一次；
    重复调用抛出 DuplicateEvaluationError，并提示已有记录的时间与元数据。

用法示例
--------
::

    from datetime import date
    from pathlib import Path
    from src.eval_protocol import WalkForwardProtocol, EvaluationLedger

    proto = WalkForwardProtocol(
        data_start=date(2018, 1, 1),
        data_end=date(2026, 8, 1),
    )
    print(proto.sterile_start)        # 2025-08-01
    for split in proto.splits:
        model = train(data[: split.train_end])
        preds = model.predict(data[split.score_start : split.score_end])

    ledger = EvaluationLedger(Path("output/eval_ledger.json"))
    ledger.record("v8_param_set_A", metadata={"sharpe": 1.2})
    ledger.record("v8_param_set_A")   # → DuplicateEvaluationError
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 内部日期工具
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """将日期向前（正数）或向后（负数）移动 months 个月，结果对齐到月末。"""
    total = d.year * 12 + (d.month - 1) + months
    year, m0 = divmod(total, 12)
    month = m0 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _quarter_start(d: date) -> date:
    """返回包含日期 d 的季度第一天（1/4/7/10 月 1 日）。"""
    month = ((d.month - 1) // 3) * 3 + 1
    return date(d.year, month, 1)


def _next_quarter_start(d: date) -> date:
    """返回 d 所在季度的下一个季度第一天。"""
    return _add_months(_quarter_start(d), 3)


def _quarter_end(q_start: date) -> date:
    """返回以 q_start 开始的季度最后一天。"""
    next_q = _add_months(q_start, 3)
    return date(next_q.year, next_q.month, 1) - _one_day()


def _one_day():
    """避免在模块级 import timedelta，用惰性方式获取。"""
    from datetime import timedelta
    return timedelta(days=1)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardSplit:
    """一次 walk-forward 切分。

    所有字段均为闭区间端点（训练数据包含 train_end 当天，打分数据包含 score_end 当天）。

    Attributes
    ----------
    train_end:
        训练窗口最后一天（含）。模型只使用 <= train_end 的数据。
    score_start:
        打分窗口第一天（含）。严格晚于 train_end。
    score_end:
        打分窗口最后一天（含）。严格早于无菌终审段起点。
    """
    train_end: date
    score_start: date
    score_end: date


# ---------------------------------------------------------------------------
# WalkForwardProtocol
# ---------------------------------------------------------------------------

class WalkForwardProtocol:
    """Walk-forward 季度滚动重训协议 + 无菌终审段管控。

    Parameters
    ----------
    data_start:
        可用数据的第一天。
    data_end:
        可用数据的最后一天（终审段截止点）。
    sterile_months:
        无菌终审段长度（月数）。默认 12，即最近 12 个月。
    min_train_quarters:
        首次打分前所需的最少训练季度数。默认 4（即最少用 1 年数据训练）。

    Attributes
    ----------
    sterile_start:
        无菌终审段起点（data_end 向前推 sterile_months 个月的月初）。
        任何训练窗口或搜索窗口均不得接触此日期及之后的数据。
    splits:
        有序的 WalkForwardSplit 列表，每项代表一次季度级重训+打分。
    """

    def __init__(
        self,
        data_start: date,
        data_end: date,
        sterile_months: int = 12,
        min_train_quarters: int = 4,
    ) -> None:
        if data_start >= data_end:
            raise ValueError(
                f"data_start ({data_start}) 必须早于 data_end ({data_end})"
            )
        if sterile_months <= 0:
            raise ValueError(f"sterile_months 必须为正整数，得到 {sterile_months}")
        if min_train_quarters <= 0:
            raise ValueError(
                f"min_train_quarters 必须为正整数，得到 {min_train_quarters}"
            )
        self._data_start = data_start
        self._data_end = data_end
        self._sterile_months = sterile_months
        self._min_train_quarters = min_train_quarters

    # ------------------------------------------------------------------
    # 核心属性
    # ------------------------------------------------------------------

    @property
    def sterile_start(self) -> date:
        """无菌终审段起点。

        取 data_end 所在月份的月初往前推 sterile_months 个月。
        例如：data_end=2026-08-15 → 月初=2026-08-01 → 推 12 个月 → 2025-08-01。
        """
        ref = date(self._data_end.year, self._data_end.month, 1)
        return _add_months(ref, -self._sterile_months)

    @property
    def splits(self) -> list[WalkForwardSplit]:
        """返回全部有效 walk-forward 切分（按时间升序）。

        生成规则
        --------
        1. 首次打分窗口起点 = data_start + min_train_quarters 个季度，
           向上对齐到季度起始日。
        2. 逐季递推：每次将打分窗口推进一个季度，同时扩展训练数据至前一季度末。
        3. 若打分窗口（即使经过截断）完全落入或超出无菌终审段，则停止生成。
        4. 若打分窗口末端超出无菌终审段起点，截断到 sterile_start - 1 天；
           截断后若 score_end < score_start，丢弃该切分。
        """
        from datetime import timedelta

        result: list[WalkForwardSplit] = []
        sterile = self.sterile_start

        # 计算最早允许的打分窗口起点
        earliest_score = _add_months(self._data_start, self._min_train_quarters * 3)

        # 找到第一个 >= earliest_score 的季度起始日
        q = _quarter_start(earliest_score)
        if q < earliest_score:
            q = _next_quarter_start(earliest_score)

        while q < sterile:
            train_end = q - timedelta(days=1)
            # train_end 不得早于 data_start
            if train_end < self._data_start:
                q = _next_quarter_start(q)
                continue

            score_start = q
            score_end_raw = _quarter_end(q)
            score_end = min(score_end_raw, sterile - timedelta(days=1))

            if score_end >= score_start:
                result.append(WalkForwardSplit(
                    train_end=train_end,
                    score_start=score_start,
                    score_end=score_end,
                ))

            q = _next_quarter_start(q)

        return result

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """断言所有切分满足协议约束，违反时抛出 ValueError。

        检查项
        ------
        - 每个切分：train_end < score_start（无前视）
        - 每个切分：score_end < sterile_start（打分窗口不触碰终审段）
        - 整体：至少存在一个有效切分
        """
        splits = self.splits
        if not splits:
            raise ValueError(
                "walk-forward 协议未生成任何有效切分，"
                "请检查 data_start/data_end/min_train_quarters 配置"
            )
        sterile = self.sterile_start
        for i, s in enumerate(splits):
            if s.train_end >= s.score_start:
                raise ValueError(
                    f"切分 #{i}：train_end ({s.train_end}) >= score_start ({s.score_start})，"
                    "训练窗口不得延伸至打分窗口"
                )
            if s.score_end >= sterile:
                raise ValueError(
                    f"切分 #{i}：score_end ({s.score_end}) >= sterile_start ({sterile})，"
                    "打分窗口侵入无菌终审段"
                )
            if s.train_end >= sterile:
                raise ValueError(
                    f"切分 #{i}：train_end ({s.train_end}) >= sterile_start ({sterile})，"
                    "训练窗口侵入无菌终审段"
                )

    def __repr__(self) -> str:
        return (
            f"WalkForwardProtocol("
            f"data_start={self._data_start}, "
            f"data_end={self._data_end}, "
            f"sterile_start={self.sterile_start}, "
            f"n_splits={len(self.splits)}, "
            f"sterile_months={self._sterile_months}, "
            f"min_train_quarters={self._min_train_quarters}"
            f")"
        )


# ---------------------------------------------------------------------------
# EvaluationLedger
# ---------------------------------------------------------------------------

class DuplicateEvaluationError(ValueError):
    """同一候选方案在终审段重复评估时抛出。

    属性 ``existing_record`` 包含首次评估的完整台账条目。
    """

    def __init__(self, message: str, existing_record: dict) -> None:
        super().__init__(message)
        self.existing_record = existing_record


class EvaluationLedger:
    """终审段评估台账：每个候选方案只允许评估一次。

    台账以 JSON 格式持久化到磁盘。
    每次 ``record()`` 调用若候选方案已存在则立即抛出 DuplicateEvaluationError。

    Parameters
    ----------
    path:
        台账 JSON 文件路径。若文件不存在，首次写入时自动创建（含父目录）。

    Thread-safety
    -------------
    本类不提供跨进程锁。单进程顺序调用是安全的；多进程并发写入需调用方自行加锁。
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._records: dict[str, dict] = self._load()

    # ------------------------------------------------------------------
    # 内部 I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, dict]:
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8")
            return json.loads(raw) if raw.strip() else {}
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._records, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def record(
        self,
        candidate_id: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """在台账中记录一次终审段评估。

        Parameters
        ----------
        candidate_id:
            候选方案唯一标识（如 "v8_param_set_A"、特征集 hash 等）。
        metadata:
            评估相关元数据（Sharpe、参数字典等），可选。

        Returns
        -------
        dict
            刚写入台账的条目。

        Raises
        ------
        DuplicateEvaluationError
            若 candidate_id 已存在于台账，立即抛出并提示已有记录。
        """
        if candidate_id in self._records:
            existing = self._records[candidate_id]
            raise DuplicateEvaluationError(
                f"候选方案 {candidate_id!r} 已于 {existing['recorded_at']} "
                f"在终审段评估过，禁止重复评估。"
                f"台账记录：{existing}",
                existing_record=existing,
            )
        entry = {
            "candidate_id": candidate_id,
            "recorded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "metadata": metadata or {},
        }
        self._records[candidate_id] = entry
        self._save()
        return entry

    def has_record(self, candidate_id: str) -> bool:
        """检查候选方案是否已有台账记录。"""
        return candidate_id in self._records

    def get_record(self, candidate_id: str) -> Optional[dict]:
        """返回指定候选方案的台账条目，若不存在则返回 None。"""
        return self._records.get(candidate_id)

    def all_records(self) -> dict[str, dict]:
        """返回全部台账条目的深拷贝。"""
        import copy
        return copy.deepcopy(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"EvaluationLedger(path={self._path!r}, n_records={len(self)})"
