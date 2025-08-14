# src/langchainagenticai/agent/agent.py
import os
import json
import re
import logging
from string import Template
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient

from .prompt_template import REACT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def _render_system_prompt(mcp_tools: List[Any]) -> str:
    tool_lines = []
    for t in mcp_tools:
        sig = ""
        if getattr(t, "input_schema", None):
            try:
                sig = json.dumps(t.input_schema, ensure_ascii=False)
            except Exception:
                sig = str(t.input_schema)
        desc = getattr(t, "description", "") or ""
        tool_lines.append(f"- {t.name}: {desc}\n  schema: {sig}")

    file_list = ", ".join(os.listdir(os.getcwd()))
    system = Template(REACT_SYSTEM_PROMPT).substitute(
        tool_list="\n".join(tool_lines),
        operating_system=os.name,
        file_list=file_list,
    )
    return system


def _coerce_arguments_for_tool(tool: Any, arguments: Any) -> Dict[str, Any]:
    """
    依据 MCP 工具的 input_schema 自动“纠正”参数形状。
    典型场景：工具要求顶层 { "input_data": {...} }，而模型给了 {...}
    """
    # 未提供 schema 就原样返回
    schema = getattr(tool, "input_schema", None)
    if not isinstance(schema, dict):
        return arguments if isinstance(arguments, dict) else {"input_data": {"query": str(arguments)}}

    required = set(schema.get("required", []))
    props = schema.get("properties", {})

    # 工具要求必须有 input_data
    needs_input_data = ("input_data" in required) or ("input_data" in props)

    # 如果需要 input_data，但当前没有，就帮忙包裹
    if needs_input_data and (not isinstance(arguments, dict) or "input_data" not in arguments):
        if isinstance(arguments, dict):
            # 常见情况：模型给了 {"query": "..."} → 包成 {"input_data": {"query": "..."}}
            return {"input_data": arguments}
        else:
            # 模型给了字符串/别的 → 尝试作为 query
            return {"input_data": {"query": str(arguments)}}

    # 否则原样返回
    return arguments if isinstance(arguments, dict) else {"input_data": {"query": str(arguments)}}


class MCPAgent:
    """
    ReAct + MCP：
    - <thought>/<action>（JSON） → 调用 MCP tool → <observation> → … → <final_answer>
    - 打印关键日志；设置最大步数避免死循环
    """

    def __init__(self, model: BaseChatModel, mcp_client: MultiServerMCPClient, max_steps: int = 8, verbose: bool = True):
        self.model = model
        self.mcp_client = mcp_client
        self.max_steps = max_steps
        self.verbose = verbose

    async def ainvoke(self, user_query: str) -> str:
        tools = await self.mcp_client.get_tools()
        if not tools:
            return "❌ 未发现任何 MCP 工具，请检查 mcp_server 是否启动、连接键名是否匹配。"

        system_prompt = _render_system_prompt(tools)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<question>{user_query}</question>"},
        ]

        used_tool_once = False
        step = 0

        while True:
            step += 1
            if step > self.max_steps:
                logger.warning("🔁 已达到最大步数上限，强制结束。")
                return "⚠️ 超过最大推理步数，模型未能在限制内完成。请重试或换个问题。"

            # 1) 调模型
            ai_msg = await self.model.ainvoke(messages)
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            messages.append({"role": "assistant", "content": content})

            # 打印 thought
            m_th = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if m_th:
                logger.info(f"[Step {step}] 💭 Thought: {m_th.group(1).strip()}")

            # 2) final_answer？
            if "<final_answer>" in content:
                if not used_tool_once:
                    warn = "错误：你尚未调用任何工具。请先输出 <action> 并调用一个 MCP 工具。"
                    logger.info(f"[Step {step}] ⛔ {warn}")
                    messages.append({"role": "user", "content": f"<observation>{warn}</observation>"})
                    continue
                m = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                final_answer = m.group(1).strip() if m else ""
                logger.info(f"[Step {step}] ✅ Final Answer: {final_answer[:200]}{'...' if len(final_answer)>200 else ''}")
                return final_answer or "（未能解析 <final_answer> 内容）"

            # 3) 解析 <action>（JSON）
            m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not m:
                err = "错误：未检测到 <action> 或 <final_answer>。请按协议输出。"
                logger.info(f"[Step {step}] ⛔ {err}")
                messages.append({"role": "user", "content": f"<observation>{err}</observation>"})
                continue

            action_block = m.group(1).strip()
            logger.info(f"[Step {step}] 🔧 Raw <action>: {action_block}")

            try:
                action_obj = json.loads(action_block)
                tool_name = action_obj["tool_name"]
                arguments = action_obj.get("arguments", {})
            except Exception as e:
                err = f"错误：<action> 必须是 JSON。解析失败：{e}"
                logger.info(f"[Step {step}] ⛔ {err}")
                messages.append({"role": "user", "content": f"<observation>{err}</observation>"})
                continue

            # 4) 匹配 MCP tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool is None:
                err = f"错误：未找到名为 {tool_name} 的 MCP 工具。"
                logger.info(f"[Step {step}] ⛔ {err}")
                messages.append({"role": "user", "content": f"<observation>{err}</observation>"})
                continue

            # ⭐ 5) **Schema 自适配**：根据 tool.input_schema 纠正参数形状（比如包裹 input_data）
            coerced_args = _coerce_arguments_for_tool(tool, arguments)
            logger.info(f"[Step {step}] 🛠️ Tool: {tool_name} | Args: {json.dumps(coerced_args, ensure_ascii=False)}")

            # 6) 调用 MCP tool
            try:
                result = await tool.ainvoke(coerced_args)
                used_tool_once = True
            except Exception as e:
                result = f"工具执行异常：{e}"

            # 7) Observation
            show = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            logger.info(f"[Step {step}] 🔎 Observation: {show[:600]}{' ...[truncated]' if len(show)>600 else ''}")

            messages.append({"role": "user", "content": f"<observation>{show}</observation>"})


async def create_mcp_agent(model: BaseChatModel, connections: Dict[str, Any]) -> MCPAgent:
    mcp_client = MultiServerMCPClient(connections=connections)
    await mcp_client.get_tools()  # 预热
    return MCPAgent(model=model, mcp_client=mcp_client, max_steps=8, verbose=True)