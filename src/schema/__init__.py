from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]


class ToolChoice(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]


class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    role: ROLE_TYPE
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    base64_image: str | None = None

    def __add__(self, other):
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other):
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    def to_langchain(self):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage as LCToolMessage,
        )

        if self.role == Role.SYSTEM:
            return SystemMessage(content=self.content or "")
        elif self.role == Role.USER:
            return HumanMessage(content=self.content or "")
        elif self.role == Role.ASSISTANT:
            if self.tool_calls:
                return AIMessage(
                    content=self.content or "",
                    tool_calls=[tc.model_dump() for tc in self.tool_calls],
                )
            return AIMessage(content=self.content or "")
        elif self.role == Role.TOOL:
            return LCToolMessage(
                content=self.content or "", tool_call_id=self.tool_call_id
            )
        return HumanMessage(content=self.content or "")

    @classmethod
    def user_message(cls, content: str, base64_image: str | None = None) -> "Message":
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: str | None = None, base64_image: str | None = None
    ) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name: str, tool_call_id: str, base64_image: str | None = None
    ) -> "Message":
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: list,
        content: str | list[str] = "",
        base64_image: str | None = None,
        **kwargs,
    ) -> "Message":
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=[ToolCall(**tc) for tc in formatted_calls],
            base64_image=base64_image,
        )


class Memory(BaseModel):
    messages: list[Message] = []
    max_messages: int = 100

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        self.messages.clear()

    def get_recent_messages(self, n: int) -> list[Message]:
        return self.messages[-n:]

    def to_dict_list(self) -> list[dict]:
        return [msg.to_dict() for msg in self.messages]
