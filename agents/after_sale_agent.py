# agents/after_sale_agent.py
# 【单 Agent】售后审批 LangGraph 图（Pipeline 串行第二棒：风险拦截 -> 售后审批）
#   节点链：START -> risk_screen(轻量风险扫描，复用三专家并行范式)
#                -> approval(interrupt 人工审批：approve / modify / reject) -> END
#   checkpointer=MemorySaver + thread_id 断点恢复；高风险必走人工
from __future__ import annotations

import asyncio
import json
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from foundation.common import PlantState, audit
from foundation.model_factory import llm_factory
from foundation.storage import save_stage


class AfterSaleDecision(BaseModel):
    action: Literal["approve", "modify", "reject"]
    reviewer_id: str = Field(min_length=1)
    comment: str = ""
    refund_amount: float | None = None


# ===================== 节点 1：轻量风险扫描（Pipeline 前序已注入 state.risk 时跳过） =====================
async def risk_screen_node(state: PlantState) -> dict:
    """
    Pipeline 串行场景：orchestrator 会先跑 rule_review 的专家面板，把 expert_findings/overall_risk
    注入 state 后再调用本图；此时直接复用，不重复评审。
    单 Agent 直达场景：state 里没有 risk，这里现场跑一次轻量扫描。
    """
    if state.get("overall_risk"):
        return audit(state, "Pipeline 前序已注入风险结果，跳过重复扫描")

    llm = llm_factory.get_llm("after-sale-risk")
    resp = await llm.ainvoke(
        f"售后工单：{state['raw_query']}\n"
        '评估退款/退货风险，只返回 JSON：{"overall_risk":"low|medium|high|critical","reason":"..."}'
    )
    data = json.loads(_strip_json(resp.content))
    return {
        "overall_risk": data["overall_risk"],
        "needs_human_review": data["overall_risk"] in ("high", "critical"),
        **audit(state, f"售后风险扫描={data['overall_risk']}"),
    }


# ===================== 节点 2：HITL 人工审批 interrupt（对齐 q5_workflow.approval_node） =====================
async def approval_node(state: PlantState) -> dict:
    # 售后退款/退货一律走人工审批（HITL interrupt）
    human = interrupt({
        "request_no": state["request_no"],
        "overall_risk": state.get("overall_risk", "high"),
        "ticket": state["raw_query"],
        "hint": "售后审批：approve / modify / reject，可携带 refund_amount",
    })
    decision = AfterSaleDecision(**human)
    await save_stage(state["request_no"], "after_sale_approval", decision.model_dump())
    return {
        "approval_status": decision.action,
        "approved_plan": json.dumps(decision.model_dump(), ensure_ascii=False),
        "result": f"售后审批完成：{decision.action}",
        **audit(state, f"售后人工审批 reviewer={decision.reviewer_id} action={decision.action}"),
    }


def _route(state: PlantState) -> Literal["done", "closed"]:
    return "closed" if state["approval_status"] == "reject" else "done"


async def done_node(state: PlantState) -> dict:
    # TODO: 售后中心写回 API 未提供，只落 stage
    return {"result": f"售后审批完成：{state['approval_status']}"}


async def closed_node(state: PlantState) -> dict:
    return {"result": "售后单已驳回"}


# ===================== 图组装 =====================
def build_graph(checkpointer=None):
    """
    【单 Agent 直达】图，也被 orchestrator.py 作为 Pipeline 第二棒复用。
    复用同一份 PlantState：orchestrator 把 rule_review 的 expert_findings/overall_risk 注入后，
    本图 risk_screen 节点会直接短路，不重复评审——这就是 Pipeline 串行"前序结构化结果注入后序上下文"。
    """
    builder = StateGraph(PlantState)
    builder.add_node("risk_screen", risk_screen_node)
    builder.add_node("approval", approval_node)
    builder.add_node("done", done_node)
    builder.add_node("closed", closed_node)

    builder.add_edge(START, "risk_screen")
    builder.add_edge("risk_screen", "approval")
    builder.add_conditional_edges("approval", _route, {"done": "done", "closed": "closed"})
    builder.add_edge("done", END)
    builder.add_edge("closed", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())


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
