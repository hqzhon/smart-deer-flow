from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, model_validator

from src.schema import AgentState, Memory, Message, ROLE_TYPE


class BaseAgent(BaseModel, ABC):
    name: str = Field(..., description="Agent name")
    description: str | None = Field(default=None, description="Agent description")
    system_prompt: str | None = Field(default=None, description="System prompt")
    next_step_prompt: str | None = Field(default=None, description="Next step prompt")
    state: AgentState = Field(default=AgentState.IDLE, description="Agent state")
    memory: Memory = Field(default_factory=Memory, description="Agent memory")
    max_steps: int = Field(default=10, description="Maximum steps")
    current_step: int = Field(default=0, description="Current step")
    duplicate_threshold: int = Field(
        default=2, description="Duplicate detection threshold"
    )

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")
        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR
            raise e
        finally:
            self.state = previous_state

    def update_memory(
        self,
        role: ROLE_TYPE,
        content: str,
        base64_image: str | None = None,
        **kwargs,
    ) -> None:
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }
        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")
        if role == "tool":
            kwargs = {"base64_image": base64_image, **kwargs}
        elif role in ("user", "assistant"):
            kwargs = {"base64_image": base64_image}
        else:
            kwargs = {}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: str | None = None) -> str:
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: list[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                step_result = await self.step()

                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        pass

    def handle_stuck_state(self):
        stuck_prompt = (
            "Observed duplicate responses. Consider new strategies and avoid "
            "repeating ineffective paths already attempted."
        )
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"

    def is_stuck(self) -> bool:
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )
        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> list[Message]:
        return self.memory.messages

    @messages.setter
    def messages(self, value: list[Message]):
        self.memory.messages = value

    def to_langchain_messages(self) -> list:
        return [msg.to_langchain() for msg in self.memory.messages]
