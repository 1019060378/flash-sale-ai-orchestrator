# 多 Agent 智能客服运营中台

![demo](demo.png)

## 系统架构

```
用户请求（运营提报 / 客诉 / 订单查询）
        ↓
unified_chat.py 【接待员 Receptionist】
├─ ① 规则前置拦截（零 Token，过滤无效、违规、重复提报）
├─ ② LLM + MiniLM 意图路由（6 类意图，映射 3 条业务分支：规则提报 / 订单履约 / 售后审批）
└─ ③ SSE 流式分发，输出 label 派活
        ↓
orchestrator.py 【调度台 Orchestrator】
├─ 单 Agent 直达：直接调用对应 LangGraph 图
└─ Pipeline 串联编排：多 Agent 串行，前序结构化结果注入后序上下文（风险拦截 → 售后审批）
        ↓
3 套独立 LangGraph Agent 图
├─ 规则上线审核 Agent（RAG 知识库检索 + 安全/停机/成本专家并行评审）
├─ 订单履约 Agent（状态机 + SSE 多轮、调用订单 API）
└─ 售后审批 Agent（HITL interrupt 人工审批，Command(resume=) 断点续跑）
        ↓
底层依赖底座（foundation/）
├─ 存储：PostgreSQL、Redis、Milvus 向量库（未配置自动降级内存模式）
├─ 模型：BGE-M3 Embedding、BGE-Reranker、LLM、MiniLM 意图分类
└─ 公共能力：三层重试降级、审计日志、SSE 封装、MCP 工具调用

应用骨架 main.py
├─ lifespan 启动：数据库迁移、模型预热、MCP 初始化
├─ 接口路由、CORS、/health 健康检查
└─ SSE 流式输出封装
```

## 快速启动

```bash
# 安装依赖
pip install fastapi uvicorn langchain-openai langgraph sentence-transformers

# 配置 .env
MODEL_API_KEY=sk-xxx
MODEL_NAME=deepseek-v4.1-flash

# 启动
python main.py
```

访问：
- 演示页：http://127.0.0.1:8000/
- 架构图：http://127.0.0.1:8000/arch
- 健康检查：http://127.0.0.1:8000/health

## 目录结构

```
flash-sale-ai-orchestrator/
├── main.py                  # FastAPI 入口 + lifespan + 路由
├── unified_chat.py          # 接待员：意图路由 + SSE 派活
├── orchestrator.py          # 调度台：单 Agent / Pipeline 编排
├── agents/
│   ├── rule_review_agent.py # 规则审核：RAG + 三专家并行 + 人工审批
│   ├── order_fulfill_agent.py # 订单履约查询
│   └── after_sale_agent.py  # 售后审批：HITL interrupt
├── foundation/
│   ├── model_factory.py     # LLM / MiniLM / BGE 模型工厂
│   ├── storage.py           # PG / Redis / Milvus（降级内存）
│   ├── common.py            # SSE / 重试 / 审计日志
│   └── mcp_client.py        # MCP 外部工具接入
├── static/
│   ├── index.html           # 前端演示页
│   └── architecture.html    # Mermaid 架构图
└── .env                     # 配置文件
```
