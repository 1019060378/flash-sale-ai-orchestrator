# foundation/mcp_client.py
# MCP 工具调用封装：订单查询、库存扣减、售后单写回等外部业务系统统一走这里。
# 业务方未提供真实 MCP Server 地址 / 工具签名，此处只保留调用协议骨架 + TODO 占位。
from __future__ import annotations

from typing import Any

from foundation.common import AgentMessage
from foundation.model_factory import get_config


class McpCallError(RuntimeError):
    pass


class McpClient:
    """
    MCP 调用统一入口：
      - 初始化时读取 .env 的 MCP_ENDPOINTS（JSON，工具名 -> 地址）
      - call_tool(tool_name, arguments) 走 HTTP/RPC 调用
      - 所有真实业务参数由上层 Agent 组装；本类不假设字段结构
    """

    def __init__(self):
        self.endpoints: dict[str, str] = {}
        self._initialized = False

    def init(self) -> None:
        # TODO: 真实环境这里应解析 .env 的 MCP_ENDPOINTS JSON，并做 MCP 握手
        raw = get_config("MCP_ENDPOINTS")
        if raw:
            import json as _json
            self.endpoints = _json.loads(raw)
        self._initialized = True

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self._initialized:
            self.init()
        if tool_name not in self.endpoints:
            # 业务未接入 MCP：直接抛占位异常，由上层 Agent 走降级分支
            raise McpCallError(
                f"MCP 工具 {tool_name} 未在 MCP_ENDPOINTS 注册；"
                f"业务侧接入后在此处补真实 HTTP/RPC 调用"
            )
        # TODO: 替换为真实 MCP 客户端 SDK 调用
        raise McpCallError(f"MCP 工具 {tool_name} 调用通道尚未实现（占位）")

    def to_message(self, sender: str, tool_name: str, result: Any) -> AgentMessage:
        return AgentMessage(
            sender=sender,
            receiver="orchestrator",
            content=f"[mcp:{tool_name}] {result}",
            msg_type="notify",
        )


mcp_client = McpClient()
