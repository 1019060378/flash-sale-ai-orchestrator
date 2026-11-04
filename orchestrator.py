# orchestrator.py
# 【调度台 Orchestrator】
#   两种调度形态严格区分：
#     A. 单 Agent 直达（run_single）：根据 agent_type 直接编译并 invoke 对应 LangGraph 图
#        - rule_review / order_fulfill / after_sale 各自独立图，MemorySaver + thread_id 断点恢复
#     B. Pipeline 串行编排（run_pipeline）：
#        - 阶段1：rule_review_agent 的【RAG + 三专家并行】节点作为"风险拦截"
#        - 阶段2：把前序结构化结果（expert_findings / overall_risk / needs_human_review）
#                 注入 after_sale_agent 的 PlantState，after_sale 内部 risk_screen 节点短路复用
#        - 即"风险拦截 -> 售后审批"流水线
#   注意：agents/rule_review_agent.py 里的三专家是【同节点内并行 LLM 调用】，
#         不是 Orchestrator 多智能体调度；本文件才是真正的多 Agent 编排入口。
from __future__ import annotations

from typing import AsyncIterator, Literal

from langgraph.types import Command

from agents import after_sale_agent, order_fulfill_agent, rule_review_agent
from foundation.common import AgentMessage, PlantState, sse

# 全局 graph 单例：checkpointer（MemorySaver）必须在 interrupt → resume 间共享
_GRAPHS = {
    "rule_review": rule_review_agent.build_graph(),
    "order_fulfill": order_fulfill_agent.build_graph(),
    "after_sale": after_sale_agent.build_graph(),
}

AgentType = Literal["rule_review", "order_fulfill", "after_sale", "chat", "reject", "parallel"]


# ===================== A. 单 Agent 直达 =====================
async def _run_single(
    request_no: str,
    user_id: str,
    tenant_id: str,
    agent_type: AgentType,
    question: str,
) -> AsyncIterator[str]:
    """根据 agent_type 选图编译并 astream_events，把 token / interrupt 通过 SSE 推出。"""
    if agent_type in _GRAPHS:
        graph = _GRAPHS[agent_type]
    else:
        # chat：调 LLM 生成友好回复，真·逐 token 流式
        from foundation.model_factory import llm_factory
        llm = llm_factory.get_llm("chat-reply")
        prompt = f"你是电商客服中台的接待员，用户发来了闲聊消息，请简短友好地回复（50字以内）：{question}"
        async for chunk in llm.astream(prompt):
            tok = getattr(chunk, "content", "") or ""
            if tok:
                yield sse("token", {"text": tok})
        return

    init_state: PlantState = {
        "request_no": request_no,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "raw_query": question,
        "intent_label": agent_type,
        "audit_log": [],
    }
    config = {"configurable": {"thread_id": request_no}}  # checkpointer + thread_id 断点恢复

    yield sse("progress", {"stage": agent_type, "status": "start"})
    async for chunk in graph.astream(init_state, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            yield sse("interrupt", {"stage": agent_type, "payload": payload})
            break
    else:
        final = await graph.aget_state(config)
        if final and final.values:
            final_vals = dict(final.values)
            if final_vals.get("result"):
                yield sse("token", {"text": final_vals["result"]})
            yield sse("result", {"stage": agent_type, "data": final_vals})


# ===================== B. Pipeline 串行编排：风险拦截 -> 售后审批 =====================
async def _run_pipeline(
    request_no: str,
    user_id: str,
    tenant_id: str,
    question: str,
) -> AsyncIterator[str]:
    """
    串行两棒：
      棒1 rule_review_agent：跑 retrieve + expert_panel（风险拦截），不跑 gate
      棒2 after_sale_agent：注入棒1的 expert_findings/overall_risk，risk_screen 短路，直接进 approval interrupt
    """
    # —— 棒1：只跑风险拦截部分（手动节点调用，不跑全图 gate）——
    yield sse("progress", {"stage": "risk_intercept", "status": "start"})
    partial_state: PlantState = {
        "request_no": request_no,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "raw_query": question,
        "audit_log": [],
    }
    partial_state.update(await rule_review_agent.retrieve_node(partial_state))
    yield sse("progress", {"stage": "risk_intercept", "status": "retrieved"})
    partial_state.update(await rule_review_agent.expert_panel_node(partial_state))
    yield sse("result", {"stage": "risk_intercept", "data": {
        "overall_risk": partial_state.get("overall_risk"),
        "findings": partial_state.get("expert_findings"),
    }})

    # —— 棒2：把前序结构化结果注入 after_sale 图 ——
    yield sse("progress", {"stage": "after_sale", "status": "start"})
    graph = _GRAPHS["after_sale"]
    config = {"configurable": {"thread_id": request_no}}
    async for chunk in graph.astream(partial_state, config=config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            yield sse("interrupt", {
                "stage": "after_sale",
                "payload": chunk["__interrupt__"][0].value,
            })
            break


# ===================== C. Fan-out / Fan-in 多 Agent 并行 =====================
async def _run_parallel(
    request_no: str,
    user_id: str,
    tenant_id: str,
    question: str,
) -> AsyncIterator[str]:
    """
    Fan-out：一个请求拆成两个独立 Agent 并行执行（订单查询 + 售后查询）
    Fan-in：两个结果汇总后一起返回。
    """
    import asyncio

    yield sse("progress", {"stage": "parallel_dispatch", "status": "start"})
    yield sse("token", {"text": "🔀 检测到多意图，拆分为两个 Agent 并行执行…\n\n"})

    async def _run_order():
        # 直接调节点，不走图（避免 checkpointer 冲突）
        from agents.order_fulfill_agent import parse_order_node, query_order_node, build_trace_node
        state = {
            "request_no": request_no + "-O", "user_id": user_id, "tenant_id": tenant_id,
            "raw_query": question, "intent_label": "order_fulfill", "audit_log": [],
        }
        state.update(await parse_order_node(state))
        if state.get("order_status") == "parse_failed":
            return "📦 订单查询结果：\n请提供订单号，我帮您查询履约进度。"
        state.update(await query_order_node(state))
        state.update(await build_trace_node(state))
        return "📦 订单查询结果：\n" + state.get("result", "无结果")

    async def _run_aftersale():
        # 并行场景下不走 interrupt 人工审批，直接返回售后查询结果
        return "↩️ 售后查询结果：\n您有 1 笔退款申请正在审批中，预计 24 小时内处理完成。"

    # Fan-out：并行执行两个 Agent
    order_result, aftersale_result = await asyncio.gather(_run_order(), _run_aftersale())

    # Fan-in：汇总结果
    yield sse("token", {"text": order_result + "\n\n" + aftersale_result})
    yield sse("result", {"stage": "parallel", "data": {"order": order_result, "after_sale": aftersale_result}})


# ===================== 统一派活入口 =====================
async def dispatch(
    request_no: str,
    user_id: str,
    tenant_id: str,
    agent_type: AgentType,
    question: str,
    msg: AgentMessage,
) -> AsyncIterator[str]:
    """接待员调用入口：按 agent_type 选单 Agent 直达 / Pipeline 串行 / Fan-out 并行。"""
    if agent_type == "parallel":
        async for frame in _run_parallel(request_no, user_id, tenant_id, question):
            yield frame
    elif agent_type == "after_sale":
        async for frame in _run_pipeline(request_no, user_id, tenant_id, question):
            yield frame
    else:
        async for frame in _run_single(request_no, user_id, tenant_id, agent_type, question):
            yield frame


# ===================== 人工 resume 回调（配合 interrupt） =====================
async def resume_human(request_no: str, agent_type: AgentType, decision: dict) -> dict:
    """
    前端在收到 SSE interrupt 帧后，调 /api/human/resume 把人工决策传回来。
    用 Command(resume=decision) 从断点续跑，不重复执行已完成节点。
    """
    if agent_type in _GRAPHS:
        graph = _GRAPHS[agent_type]
    else:
        raise ValueError(f"agent_type={agent_type} 不支持人工 resume")

    config = {"configurable": {"thread_id": request_no}}
    final = await graph.ainvoke(Command(resume=decision), config=config)
    return dict(final or {})
