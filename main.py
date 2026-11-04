# main.py
# FastAPI 应用骨架
#   lifespan：数据库 init_db / 本地模型预热（MiniLM、BGE-M3、Reranker）/ MCP 初始化
#   路由：
#     GET  /health               健康检查
#     POST /api/chat/stream      SSE 统一入口（接待员 -> 调度台 -> 各 Agent 图）
#     POST /api/human/resume     人工审批 interrupt 恢复（Command(resume=...)）
#   CORS、SSE 统一封装在 foundation.common.sse
from __future__ import annotations

import asyncio
import warnings
from contextlib import asynccontextmanager

# 屏蔽 requests 与 urllib3/charset_normalizer 版本不匹配的无害依赖警告
warnings.filterwarnings(
    "ignore",
    message="urllib3 .* or chardet .* doesn't match a supported version",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from foundation.mcp_client import mcp_client
from foundation.model_factory import bge_embedder, bge_reranker, minilm_classifier
from foundation.storage import init_db
from orchestrator import resume_human
from unified_chat import chat_stream


# ===================== Lifespan：启动时建表 + 预热模型 + 初始化 MCP =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 数据库迁移（建表）
    await init_db()

    # 2) MCP 客户端初始化（未配置端点时静默跳过）
    mcp_client.init()

    # 3) 本地模型预热：放后台任务，不阻塞启动；未配置则静默跳过，加载失败才提示
    async def _warmup():
        from foundation.model_factory import get_config
        warmup_targets = [
            ("MiniLM", "MINILM_MODEL", minilm_classifier._ensure_loaded),
            ("BGE-M3", "BGE_EMBED_MODEL", bge_embedder._ensure_loaded),
            ("BGE-Reranker", "BGE_RERANK_MODEL", bge_reranker._ensure_loaded),
        ]
        for name, env_key, loader in warmup_targets:
            if not get_config(env_key):
                continue  # 未配置路径：静默跳过，路由时会按异常轨降级到 LLM
            try:
                await asyncio.to_thread(loader)
            except Exception as exc:
                # 配了路径但模型损坏/依赖缺失：值得知道，不崩服务
                print(f"[warmup] {name} 预热失败: {exc.__class__.__name__}: {exc}")

    asyncio.create_task(_warmup())

    yield


app = FastAPI(title="flash-sale-ai-orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 路由 =====================
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def demo_page():
    """演示用前端页面：直接访问 http://127.0.0.1:8000/"""
    from pathlib import Path
    from fastapi.responses import HTMLResponse
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/arch", include_in_schema=False)
async def arch_page():
    """系统架构图：http://127.0.0.1:8000/arch"""
    from pathlib import Path
    from fastapi.responses import HTMLResponse
    html = (Path(__file__).parent / "static" / "architecture.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


class ChatRequest(BaseModel):
    question: str
    user_id: str = "u-default"
    tenant_id: str = "tenant-default"


@app.post("/api/chat/stream")
async def chat_stream_api(req: ChatRequest):
    """SSE 统一入口：接待员零 Token 拦截 + MiniLM/LLM 路由 + 调度台派活。"""
    return StreamingResponse(
        chat_stream(req.question, req.user_id, req.tenant_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ResumeRequest(BaseModel):
    request_no: str
    agent_type: str          # rule_review / after_sale
    decision: dict           # ReviewDecision / AfterSaleDecision 的 dict 形式


@app.post("/api/human/resume")
async def human_resume(req: ResumeRequest):
    """人工审批回调：用 Command(resume=decision) 从 LangGraph interrupt 断点续跑。"""
    try:
        final = await resume_human(req.request_no, req.agent_type, req.decision)
        return {"ok": True, "state": final}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
