# agents/rule_review_agent.py
# 【单 Agent】规则上线审核 LangGraph 图
#   节点链：START -> retrieve(RAG: BGE-M3 dense+sparse + RRF + Rerank + HyDE/MultiQuery)
#                -> expert_panel(安全/成本/体验 三专家【并行 LLM 调用，非 Orchestrator 调度】)
#                -> gate(低风险自动通过 / 高风险 interrupt HITL 人工复核) -> END
#   checkpointer=MemorySaver，thread_id 做断点恢复；interrupt 后用 Command(resume=...) 续跑
from __future__ import annotations

import asyncio
import json
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from foundation.common import AgentMessage, PlantState, audit
from foundation.model_factory import bge_embedder, bge_reranker, llm_factory
from foundation.storage import milvus_store, save_stage

COLLECTION = "rule_manuals"


# ===================== 结构化输出 Schema =====================
class RetrievalPackage(BaseModel):
    docs: list[dict] = Field(default_factory=list)
    advice: str = ""


class ExpertFinding(BaseModel):
    expert: Literal["安全专家", "成本专家", "体验专家"]
    risk_level: Literal["low", "medium", "high", "critical"]
    reason: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)
    is_fallback: bool = False


class ReviewDecision(BaseModel):
    action: Literal["approve", "modify", "reject"]
    reviewer_id: str = Field(min_length=1)
    comment: str = ""
    modified_rule: str | None = None


# ===================== 节点 1：RAG 检索（BGE-M3 dense+sparse，RRF 融合，Rerank 重排） =====================
async def retrieve_node(state: PlantState) -> dict:
    """
    RAG 链路对齐 q3_retrieval：
      - HyDE：先用 LLM 把模糊 query 改写成假设性规则文档
      - Multi-Query：LLM 并发拆 3 条子查询
      - BGE-M3 dense + sparse 双路召回
      - RRF（Reciprocal Rank Fusion）融合两路结果
      - BGE-Reranker 重排取 Top3
    """
    raw = state["raw_query"]
    loop = asyncio.get_running_loop()

    # ① HyDE / Multi-Query 改写
    llm = llm_factory.get_llm("rule-retriever")
    rewrite_resp = await llm.ainvoke(
        "把下面这条规则提报改写成 1 段假设性规则文档 + 3 条子检索 query，JSON 返回："
        '{"hyde":"...","queries":["...","...","..."]}\n'
        f"规则：{raw}"
    )
    rew = json.loads(_strip_json(rewrite_resp.content))
    candidates = list({rew["hyde"], *rew["queries"], raw})

    # ② 并发多路向量检索（同步 Milvus 调用放线程池）
    def _search_one(q: str) -> list[dict]:
        try:
            emb = bge_embedder.encode([q])
            from pymilvus import AnnSearchRequest, WeightedRanker
            dense_req = AnnSearchRequest([emb["dense_vecs"][0]], "dense_vector", {"metric_type": "COSINE"}, 4)
            sparse_req = AnnSearchRequest([emb["lexical_weights"][0]], "sparse_vector", {"metric_type": "IP"}, 4)
            hits = milvus_store.client.hybrid_search(
                COLLECTION, [dense_req, sparse_req],
                WeightedRanker(0.7, 0.3), limit=4,
                output_fields=["doc_id", "title", "content"],
            )[0]
            return [{"doc_id": h["entity"]["doc_id"], "title": h["entity"]["title"],
                     "content": h["entity"]["content"], "score": float(h["distance"])} for h in hits]
        except Exception:
            # BGE 未配置 / Milvus 未起 / 集合不存在：降级为空结果
            return []

    groups = await asyncio.gather(*[loop.run_in_executor(None, _search_one, q) for q in candidates])

    # ③ RRF 融合
    rrf_map: dict[str, dict] = {}
    for g in groups:
        for rank, doc in enumerate(g, start=1):
            key = doc["doc_id"]
            rrf_score = 1.0 / (60 + rank)
            if key not in rrf_map:
                rrf_map[key] = {**doc, "rrf_score": 0.0}
            rrf_map[key]["rrf_score"] += rrf_score
    fused = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)[:8]

    # ④ Rerank 重排取 Top3
    if fused:
        scores = bge_reranker.predict(raw, [d["content"] for d in fused])
        for d, s in zip(fused, scores):
            d["final_score"] = s
        fused = sorted(fused, key=lambda x: x["final_score"], reverse=True)[:3]

    return {
        "retrieval_docs": fused,
        "retrieval_advice": "\n".join(f"【{d['title']}】{d['content']}" for d in fused),
        **audit(state, "RAG 检索完成（HyDE+MultiQuery+RRF+Rerank）"),
    }


# ===================== 节点 2：三专家并行评审（普通 LLM 并行调用，非 Orchestrator 调度） =====================
EXPERTS = ("安全专家", "成本专家", "体验专家")
RISK_ORDER = ["low", "medium", "high", "critical"]


async def expert_panel_node(state: PlantState) -> dict:
    """
    【普通并行 LLM 调用】asyncio.gather 并发三位专家，单个失败降级为 fallback（对齐 q4_risk）。
    注意：这不是 Orchestrator 多智能体调度——这里是同一个 LangGraph 节点内部并发 LLM，
    Agent 间不通过 AgentMessage 通信；Orchestrator 多智能体调度见 orchestrator.py。
    """
    async def call_one(name: str) -> ExpertFinding:
        prompt = (
            f"你是大促规则审核的{name}。规则内容：{state['raw_query']}\n"
            f"参考知识库：{state.get('retrieval_advice', '')}\n"
            '只返回 JSON：{"risk_level":"low|medium|high|critical","reason":"...","suggestion":"..."}'
        )
        llm = llm_factory.get_llm(f"rule-{name}")
        resp = await llm.ainvoke(prompt)
        data = json.loads(_strip_json(resp.content))
        return ExpertFinding(expert=name, **data)

    results = await asyncio.gather(*[call_one(n) for n in EXPERTS], return_exceptions=True)
    findings: list[ExpertFinding] = []
    has_fallback = False
    for name, res in zip(EXPERTS, results):
        if isinstance(res, Exception):
            findings.append(ExpertFinding(
                expert=name, risk_level="high",
                reason=f"{name} 调用异常", suggestion="转人工复核", is_fallback=True,
            ))
            has_fallback = True
        else:
            findings.append(res)

    max_idx = max(RISK_ORDER.index(f.risk_level) for f in findings)
    overall = RISK_ORDER[max_idx]
    return {
        "expert_findings": [f.model_dump() for f in findings],
        "overall_risk": overall,
        "needs_human_review": overall in ("medium", "high", "critical") or has_fallback,
        **audit(state, f"三专家评审完成，overall_risk={overall}"),
    }


# ===================== 节点 3：汇总评审结论（评分 + 问题 + 能否上线） =====================
async def gate_node(state: PlantState) -> dict:
    """LLM 汇总三专家意见，输出最终评分、问题清单和上线结论。"""
    findings = state.get("expert_findings", [])
    findings_text = "\n".join(
        f"【{f['expert']}】风险={f['risk_level']}：{f['reason']}\n建议：{f['suggestion']}"
        for f in findings
    )
    llm = llm_factory.get_llm("rule-gate")
    resp = await llm.ainvoke(
        f"你是大促规则审核的终审官。基于以下三专家评审意见，给出最终结论。\n\n"
        f"规则：{state['raw_query']}\n\n"
        f"{findings_text}\n\n"
        f"请严格按 JSON 返回："
        f'{{"score":0-100整数,"problems":["问题1","问题2"],"conclusion":"一句话总结"}}'
        f"\n注意：85分以上可上线，低于85分不可上线。"
    )
    try:
        result = json.loads(_strip_json(resp.content))
    except Exception:
        result = {"score": 0, "problems": ["终审解析失败"], "conclusion": "需人工复核"}

    score = result.get("score", 0)
    can_online = score >= 85
    problems = result.get("problems", [])
    conclusion = result.get("conclusion", "")

    status = "approve" if can_online else "reject"
    result_text = (
        f"终审评分：{score}分 | {'✅ 可上线' if can_online else '❌ 不可上线'}\n"
        f"问题清单：{'；'.join(problems) if problems else '无'}\n"
        f"总结：{conclusion}"
    )
    return {
        "approval_status": status,
        "approved_plan": state["raw_query"] if can_online else None,
        "result": result_text,
        **audit(state, f"终审 score={score} can_online={can_online}"),
    }


def route_after_gate(state: PlantState) -> Literal["done", "closed"]:
    return "closed" if state["approval_status"] == "reject" else "done"


async def done_node(state: PlantState) -> dict:
    # TODO: 业务侧未提供规则上线写回 API，此处只落库 stage
    await save_stage(state["request_no"], "rule_review", {
        "status": state["approval_status"],
        "approved_plan": state.get("approved_plan"),
        "findings": state.get("expert_findings"),
    })
    return {}  # 不覆盖 gate_node 的 result


async def closed_node(state: PlantState) -> dict:
    await save_stage(state["request_no"], "rule_review", {"status": "rejected"})
    return {}  # 不覆盖 gate_node 的 result


# ===================== 图组装：checkpointer + MemorySaver + thread_id 断点恢复 =====================
def build_graph(checkpointer=None):
    """
    【单 Agent 直达】图。
    checkpointer 默认用 MemorySaver；调用方传入 config={"configurable":{"thread_id": request_no}}
    即可在 interrupt 暂停后，用 Command(resume=decision) 从断点续跑，不重复执行 retrieve/expert_panel。
    """
    builder = StateGraph(PlantState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("expert_panel", expert_panel_node)
    builder.add_node("gate", gate_node)
    builder.add_node("done", done_node)
    builder.add_node("closed", closed_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "expert_panel")
    builder.add_edge("expert_panel", "gate")
    builder.add_conditional_edges("gate", route_after_gate, {"done": "done", "closed": "closed"})
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
