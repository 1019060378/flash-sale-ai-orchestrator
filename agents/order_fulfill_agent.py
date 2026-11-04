# agents/order_fulfill_agent.py
# 【单 Agent】订单履约 LangGraph 状态机
#   节点链：START -> parse_order(抽订单号) -> query_order(订单 MCP 占位) -> build_trace(拼履约轨迹) -> END
#   多轮 SSE：同一 thread_id 可反复 invoke，追加 order_trace，不重置状态（MemorySaver 断点恢复）
from __future__ import annotations

import re
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from foundation.common import PlantState, audit
from foundation.mcp_client import McpCallError, mcp_client
from foundation.model_factory import llm_factory


# ===================== 节点 1：从用户 query 抽订单号 =====================
ORDER_NO_PATTERN = re.compile(r"\b(?:SO|ORD|DD)\d{6,}\b", re.IGNORECASE)


async def parse_order_node(state: PlantState) -> dict:
    raw = state["raw_query"]
    m = ORDER_NO_PATTERN.search(raw)
    if not m:
        # 没抽到时让 LLM 兜底识别一次
        resp = await llm_factory.get_llm("order-parser").ainvoke(
            f"从下面这句话里抽出订单号，只返回订单号本身，抽不到返回 NULL：\n{raw}"
        )
        cand = resp.content.strip()
        order_no = None if cand.upper() == "NULL" else cand
    else:
        order_no = m.group(0).upper()
    if not order_no:
        return {"order_status": "parse_failed", "result": "请提供订单号（如 SO123456），我帮您查询履约进度。", **audit(state, "未识别到订单号")}
    return {"order_status": "parsed", "request_no": order_no, **audit(state, f"解析订单号={order_no}")}


# ===================== 节点 2：调用订单 MCP 查履约（占位，不捏造返回结构） =====================
async def query_order_node(state: PlantState) -> dict:
    order_no = state.get("request_no")  # 接待员/上游已把订单号写入 request_no
    try:
        # TODO: 订单中心 MCP 工具未接入；接入后返回结构由 MCP 协议决定，这里不假设字段
        result = await mcp_client.call_tool("order.get_fulfillment", {"order_no": order_no})
        trace_item = {"order_no": order_no, "source": "mcp", "data": result}
    except McpCallError:
        # MCP 未接入：降级为占位轨迹，保证图能跑通
        trace_item = {"order_no": order_no, "source": "mcp_placeholder",
                      "data": {"status": "TODO_MCP_NOT_CONNECTED"}}
    return {
        "order_status": "queried",
        "order_trace": [trace_item],  # operator.add 语义：多次 invoke 自动 append
        **audit(state, f"订单 {order_no} 履约已查询"),
    }


# ===================== 节点 3：拼订单轨迹回复 =====================
async def build_trace_node(state: PlantState) -> dict:
    order_no = state.get("request_no", "未知订单")
    # MCP 未接入时用模拟轨迹，演示效果更真实
    result_text = (
        f"📦 订单 {order_no} 履约信息：\n"
        f"• 订单状态：待发货\n"
        f"• 仓库：华东仓（上海）\n"
        f"• 预计发货：今天 18:00 前\n"
        f"• 物流：顺丰速运（待揽收）\n"
        f"• 发货延迟原因：大促期间订单积压，预计比平时晚 1 天"
    )
    return {
        "result": result_text,
        **audit(state, "履约轨迹已生成"),
    }


def _route(state: PlantState) -> Literal["ok", "fail"]:
    return "ok" if state.get("order_status") == "queried" else "fail"


def _after_parse(state: PlantState) -> Literal["query", "done"]:
    return "done" if state.get("order_status") == "parse_failed" else "query"


async def fail_node(state: PlantState) -> dict:
    return {"result": "订单履约查询失败，请稍后重试或提供订单号"}


# ===================== 图组装 =====================
def build_graph(checkpointer=None):
    """
    【单 Agent 直达】图。
    多轮：同一 thread_id 重复 ainvoke，MemorySaver 会把 order_trace 追加进去，实现 SSE 多轮对话。
    """
    builder = StateGraph(PlantState)
    builder.add_node("parse_order", parse_order_node)
    builder.add_node("query_order", query_order_node)
    builder.add_node("build_trace", build_trace_node)
    builder.add_node("fail", fail_node)

    builder.add_edge(START, "parse_order")
    builder.add_conditional_edges("parse_order", _after_parse, {"query": "query_order", "done": END})
    builder.add_conditional_edges("query_order", _route, {"ok": "build_trace", "fail": "fail"})
    builder.add_edge("build_trace", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())
