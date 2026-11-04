# foundation/storage.py
# 存储封装：PostgreSQL（SQLAlchemy 异步，对齐 q2_storage）+ Redis + Milvus
# 外部连接串从 .env 读取；未配置 / 缺驱动时自动降级为内存 no-op，保证服务能起来
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from foundation.model_factory import get_config


# ===================== PostgreSQL（懒加载 + 缺驱动降级到内存） =====================
# 默认空：.env 不配 PG_URL 时直接走内存模式，不尝试连接、不打印降级噪音
PG_URL = get_config("PG_URL", "")

_pg_engine = None
_async_session_local = None
_pg_available: Optional[bool] = None

# 内存兜底：key=(request_no, stage) -> payload dict
_mem_stage: dict[tuple[str, str], dict] = {}


def _ensure_pg():
    """首次访问时才建 PG 引擎；未配置 / 缺驱动 / 连不上则降级为内存模式。"""
    global _pg_engine, _async_session_local, _pg_available
    if _pg_available is not None:
        return

    # .env 没配 PG_URL：直接走内存，不尝试建引擎
    if not PG_URL:
        _pg_available = False
        return

    try:
        from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
        _pg_engine = create_async_engine(PG_URL, pool_pre_ping=True)
        _async_session_local = async_sessionmaker(_pg_engine, expire_on_commit=False)
        _pg_available = True
    except Exception as exc:
        # 配了 PG_URL 但缺 asyncpg / 连不上：提示降级（用户主动配了却没生效，值得知道）
        print(f"[storage] PostgreSQL 不可用，降级为内存模式：{exc.__class__.__name__}: {exc}")
        _pg_available = False


async def init_db() -> None:
    """建表。PG 不可用时跳过（内存模式无需建表）。"""
    _ensure_pg()
    if not _pg_available:
        return
    from sqlalchemy import text
    async with _async_session_local() as session:
        async with session.begin():
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    request_no TEXT NOT NULL,
                    intent_label TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    UNIQUE (tenant_id, request_no)
                )
            """))
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS stage_results (
                    request_no TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (request_no, stage)
                )
            """))


async def save_stage(request_no: str, stage: str, payload: dict) -> None:
    _ensure_pg()
    if not _pg_available:
        _mem_stage[(request_no, stage)] = payload
        return
    from sqlalchemy import text
    payload_json = json.dumps(payload, ensure_ascii=False)
    updated_at = datetime.now().isoformat()
    async with _async_session_local() as session:
        async with session.begin():
            row = (await session.execute(
                text("SELECT 1 FROM stage_results WHERE request_no=:rn AND stage=:st"),
                {"rn": request_no, "st": stage},
            )).first()
            if row:
                await session.execute(text("""
                    UPDATE stage_results SET payload_json=:pj, updated_at=:ua
                    WHERE request_no=:rn AND stage=:st
                """), {"pj": payload_json, "ua": updated_at, "rn": request_no, "st": stage})
            else:
                await session.execute(text("""
                    INSERT INTO stage_results (request_no, stage, payload_json, updated_at)
                    VALUES (:rn, :st, :pj, :ua)
                """), {"rn": request_no, "st": stage, "pj": payload_json, "ua": updated_at})


async def get_stage(request_no: str, stage: str) -> Optional[dict]:
    _ensure_pg()
    if not _pg_available:
        return _mem_stage.get((request_no, stage))
    from sqlalchemy import text
    async with _async_session_local() as session:
        row = (await session.execute(
            text("SELECT payload_json FROM stage_results WHERE request_no=:rn AND stage=:st"),
            {"rn": request_no, "st": stage},
        )).mappings().first()
    return json.loads(row["payload_json"]) if row else None


async def get_all_stages(request_no: str) -> dict[str, dict]:
    _ensure_pg()
    if not _pg_available:
        return {stage: payload for (rn, stage), payload in _mem_stage.items() if rn == request_no}
    from sqlalchemy import text
    async with _async_session_local() as session:
        rows = (await session.execute(
            text("SELECT stage, payload_json FROM stage_results WHERE request_no=:rn"),
            {"rn": request_no},
        )).mappings().all()
    return {r["stage"]: json.loads(r["payload_json"]) for r in rows}


# ===================== Redis（会话幂等 / 防重复提报 / thread 缓存） =====================
class RedisClient:
    def __init__(self):
        self._client = None
        self._tried = False  # 是否已尝试过初始化（避免每次调用都重试）
        self._mem: dict[str, str] = {}
        self._mem_ttl: dict[str, float] = {}

    def _ensure(self):
        if self._tried:
            return
        self._tried = True
        url = get_config("REDIS_URL", "")
        if not url:
            self._client = None  # .env 没配：直接走内存
            return
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(url, decode_responses=True)
        except Exception:
            self._client = None  # 没装 redis-py：降级内存

    async def get(self, key: str) -> Optional[str]:
        self._ensure()
        if self._client is None:
            # 简单 TTL 过期
            import time
            expire_at = self._mem_ttl.get(key, 0)
            if expire_at and time.time() > expire_at:
                self._mem.pop(key, None)
                self._mem_ttl.pop(key, None)
                return None
            return self._mem.get(key)
        return await self._client.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._ensure()
        if self._client is None:
            import time
            self._mem[key] = value
            self._mem_ttl[key] = time.time() + ttl
            return
        await self._client.setex(key, ttl, value)


redis_client = RedisClient()


# ===================== Milvus（RAG 向量库，对齐 q3_retrieval.KnowledgeBaseClient） =====================
class MilvusStore:
    def __init__(self):
        self._client = None
        self.uri = get_config("MILVUS_URI", "./flash_sale_vectors.db")

    def _ensure(self):
        if self._client is not None:
            return
        from pymilvus import MilvusClient
        self._client = MilvusClient(uri=self.uri, token=get_config("MILVUS_TOKEN") or None)

    @property
    def client(self):
        self._ensure()
        return self._client


milvus_store = MilvusStore()
