# foundation/common.py
# 公共组件：AgentMessage 通信载体、PlantState / EventState 状态、SSE 封装、三层重试降级、审计日志
# 风格对齐参考仓库 agent项目/q4_orchestrator_pipeline.py、q5_workflow.py、q6_stream.py
from __future__ import annotations

import asyncio
import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, AsyncIterator, Literal, Optional, TypedDict


# ===================== Agent 间统一通信消息（对齐 q4_orchestrator_pipeline.AgentMessage） =====================
@dataclass
class AgentMessage:
    sender: str                     # 发送方 Agent 名
    receiver: Optional[str]         # 接收方，None=广播给 Orchestrator
    content: str
    msg_type: Literal["task", "result", "notify", "interrupt"] = "task"


# ===================== 业务主状态：规则审核 / 售后审批流水线（对齐 q5_workflow.PlantState） =====================
class PlantState(TypedDict, total=False):
    request_no: str                 # 业务单号（规则提报单号 / 售后工单 / 订单号）
    user_id: str
    tenant_id: str
    # —— 接待员产物 ——
    raw_query: str                  # 用户原始输入
    intent_label: str               # 接待员打标：rule_review / order_fulfill / after_sale
    intent_reason: str
    # —— RAG 阶段 ——
    retrieval_docs: list[dict]
    retrieval_advice: str
    # —— 多专家并行评审阶段（风险拦截） ——
    expert_findings: list[dict]
    overall_risk: str
    needs_human_review: bool
    # —— 方案 / 审批阶段 ——
    proposal: str
    approval_status: str            # approve / modify / reject / auto_approve
    approved_plan: Optional[str]
    # —— 订单履约阶段 ——
    order_status: str
    order_trace: list[dict]
    # —— 通用 ——
    audit_log: Annotated[list[str], operator.add]   # 多节点 append 合并，对齐 q5
    result: str


# ===================== 接待员事件状态（对齐 q6_stream.EventState） =====================
class EventState(TypedDict, total=False):
    question: str
    route: str                      # reject / rule_review / order_fulfill / after_sale / chat
    reason: str
    answer: str
    answer_mode: str


# ===================== SSE 统一封装（对齐 q6_stream.sse） =====================
def sse(event: str, data: dict) -> str:
    """统一 SSE 帧：event: xxx\\ndata: {json}\\n\\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_wrap(agen: AsyncIterator[str]) -> AsyncIterator[str]:
    """SSE 生成器兜底：异常帧 + 唯一 done。对齐 q6_stream.stream_incident 的 finally 语义。"""
    try:
        async for frame in agen:
            yield frame
    except Exception as exc:  # 三层兜底：最外层
        yield sse("error", {"msg": f"服务内部异常: {exc.__class__.__name__}"})
    finally:
        yield sse("done", {})


# ===================== 三层重试降级（对齐 q1_model.LLMFactory.RETRYABLE 思想） =====================
RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
)


async def with_retry(
    coro_factory,
    *,
    retries: int = 2,
    base_delay: float = 0.5,
    fallback=None,
):
    """
    三层重试降级：
      第 1 层：同步重试 retries 次，指数退避；
      第 2 层：仍失败走 fallback 回调（若提供）；
      第 3 层：fallback 也没有则抛 ModelCallError（由上层 SSE 兜底帧捕获）。
    """
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await coro_factory()
        except RETRYABLE_EXC as exc:
            last_exc = exc
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(base_delay * attempt)
    if fallback is not None:
        return fallback()
    raise RuntimeError(f"三层重试耗尽: {last_exc}") from last_exc


# ===================== 审计日志工具（对齐 PlantState.audit_log append 语义） =====================
def audit(state: PlantState, message: str) -> dict:
    """节点返回值里塞一段 audit_log，operator.add 会自动合并到主状态。"""
    return {"audit_log": [f"[{asyncio.get_event_loop().time():.2f}] {message}"]}
