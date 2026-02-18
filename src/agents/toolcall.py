import asyncio
import json
from typing import Any

from pydantic import Field

from src.agents.react import ReActAgent
from src.schema import AgentState, Message, ToolCall, ToolChoice
from src.tools.tool_collection import ToolCollection

TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    name: str = "toolcall"
    description: str = "An agent that can execute tool calls."
    system_prompt: str = ""
    next_step_prompt: str = ""

    available_tools: ToolCollection = Field(default_factory=ToolCollection)
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    _current_base64_image: str | None = None
    _llm_instance: Any = None

    max_steps: int = 30
    max_observe: int | bool | None = None

    def get_llm(self):
        if self._llm_instance is None:
            from src.llms.llm import get_llm_by_type

            self._llm_instance = get_llm_by_type("basic")
        return self._llm_instance

    async def think(self) -> bool:
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            llm = self.get_llm()
            lc_messages = self.to_langchain_messages()

            if self.system_prompt:
                from langchain_core.messages import SystemMessage

                lc_messages = [SystemMessage(content=self.system_prompt)] + lc_messages

            tool_schemas = self.available_tools.to_params()

            response = await llm.ainvoke(
                lc_messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice=self.tool_choices.value if tool_schemas else None,
            )

            tool_calls = []
            content = ""

            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type="function",
                            function={
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        )
                    )
            if hasattr(response, "content"):
                content = response.content or ""

            self.tool_calls = tool_calls

            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    pass
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            assistant_msg = (
                Message.from_tool_calls(tool_calls=self.tool_calls, content=content)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True

            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            self._current_base64_image = None
            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            args = json.loads(command.function.arguments or "{}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            await self._handle_special_tool(name=name, result=result)

            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image

            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )
            return observation
        except json.JSONDecodeError:
            return f"Error parsing arguments for {name}: Invalid JSON format"
        except Exception as e:
            return f"Error: Tool '{name}' encountered a problem: {str(e)}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        return True

    def _is_special_tool(self, name: str) -> bool:
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    await tool_instance.cleanup()
                except Exception:
                    pass

    async def run(self, request: str | None = None) -> str:
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
