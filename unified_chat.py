# unified_chat.py
# 【接待员】统一聊天入口
#   职责：
#     1) 零 Token 前置拦截：空串 / 违规词 / Redis 重复提报（幂等），全程不调 LLM、不调 MiniLM
#     2) MiniLM 意图路由：本地模型 embedding 命中 6 类意图阈值即打 label（零 Token）
#        未命中再走 LLM 兜底分类
#     3) label 映射：rule_review / order_fulfill / after_sale / chat / reject
#     4) SSE 流式派活：先推 meta(route)，再把任务交给 orchestrator
#   状态：EventState（对齐 q6_stream）；AgentMessage 用于和 orchestrator 通信
from __future__ import annotations

import json
import uuid
from typing import AsyncIterator, Literal

from pydantic import BaseModel, Field

from foundation.common import AgentMessage, EventState, sse, sse_wrap
from foundation.model_factory import llm_factory, minilm_classifier
from foundation.storage import redis_client
from orchestrator import dispatch

# 6 类意图 -> 3 个业务 label 的映射表
INTENT_TO_LABEL: dict[str, Literal["rule_review", "order_fulfill", "after_sale", "chat", "reject"]] = {
    "rule_review": "rule_review",
    "order_fulfill": "order_fulfill",
    "after_sale": "after_sale",
    "chat": "chat",
    "reject": "reject",
}

# 零 Token 违规词表（对齐 q6_stream.prefilter）
FORBIDDEN_WORDS = ("忽略所有规则", "删除数据库", "输出api key", "system prompt")


class RouteDecision(BaseModel):
    label: Literal["rule_review", "order_fulfill", "after_sale", "chat", "reject"]
    reason: str = Field(min_length=1)
    source: Literal["rule", "minilm", "llm"] = "rule"


# ===================== 零 Token 前置拦截（不调任何模型） =====================
async def zero_token_prefilter(question: str) -> RouteDecision | None:
    q = question.strip()
    if not q:
        return RouteDecision(label="reject", reason="问题不能为空", source="rule")
    if any(w in q.lower() for w in FORBIDDEN_WORDS):
        return RouteDecision(label="reject", reason="命中违规词", source="rule")

    # Redis 幂等：同一用户 5 秒内重复提报直接拦截
    dedup_key = f"chat:dedup:{uuid.uuid5(uuid.NAMESPACE_DNS, q).hex}"
    if await redis_client.get(dedup_key):
        return RouteDecision(label="reject", reason="重复提报（5 秒内）", source="rule")
    await redis_client.setex(dedup_key, ttl=5, value="1")
    return None


# ===================== MiniLM 本地意图分类（零 Token 命中） =====================
def minilm_route(question: str) -> RouteDecision | None:
    try:
        pred = minilm_classifier.predict(question)
    except Exception:
        # 模型未加载 / 未配置：静默降级到 LLM 路由
        return None
    if pred is None:
        return None
    return RouteDecision(label=INTENT_TO_LABEL.get(pred.label, "chat"),
                         reason=f"MiniLM 命中 score={pred.score:.3f}",
                         source="minilm")


# ===================== LLM 兜底路由 =====================
async def llm_route(question: str) -> RouteDecision:
    llm = llm_factory.get_llm("receptionist-router")
    resp = await llm.ainvoke(
        "你是接待员路由器。把用户问题分类为以下之一："
        "rule_review(规则/活动规则审核提报)、order_fulfill(订单/发货/物流查询)、"
        "after_sale(退款/退货/客诉/售后审批)、parallel(同时提到订单+退款/售后等多个意图)、"
        "chat(闲聊)、reject(违规)。"
        '只返回 JSON：{"label":"...","reason":"..."}\n'
        f"用户问题：{question}"
    )
    data = json.loads(_strip_json(resp.content))
    return RouteDecision(label=data["label"], reason=data["reason"], source="llm")


# ===================== SSE 派活主流程 =====================
async def chat_stream(question: str, user_id: str, tenant_id: str) -> AsyncIterator[str]:
    """接待员主入口：拦截 -> 路由 -> 派活 -> SSE。"""
    request_no = f"CHAT-{uuid.uuid4().hex[:10]}"

    async def _gen() -> AsyncIterator[str]:
        # 1) 先推 meta
        yield sse("meta", {"request_no": request_no, "user_id": user_id})

        # 2) 零 Token 前置拦截
        blocked = await zero_token_prefilter(question)
        if blocked is not None:
            yield sse("route", {"label": blocked.label, "reason": blocked.reason, "source": blocked.source})
            yield sse("result", {"stage": "answer", "data": {"content": f"请求被拦截：{blocked.reason}"}})
            return

        # 3) MiniLM 本地路由
        decision = minilm_route(question)
        # 4) LLM 兜底
        if decision is None:
            decision = await llm_route(question)

        yield sse("route", {"label": decision.label, "reason": decision.reason, "source": decision.source})

        # 5) 派活给 orchestrator
        msg = AgentMessage(
            sender="receptionist",
            receiver=decision.label,
            content=question,
            msg_type="task",
        )
        async for frame in dispatch(request_no, user_id, tenant_id, decision.label, question, msg):
            yield frame

    async for frame in sse_wrap(_gen()):
        yield frame


def _strip_json(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s[3:].lstrip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    if s.endswith("```"):
        s = s[:-3].rstrip()
    a, b = s.find("{"), s.rfind("}")
    return s[a:b + 1] if a >= 0 and b >= 0 else s
