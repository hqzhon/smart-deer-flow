import pytest

from src.agents.base import BaseAgent
from src.agents.react import ReActAgent
from src.schema import AgentState, Message, Memory


class ConcreteAgent(BaseAgent):
    name: str = "test_agent"
    description: str = "A test agent"
    step_count: int = 0

    async def step(self) -> str:
        self.step_count += 1
        if self.step_count >= 3:
            self.state = AgentState.FINISHED
        return f"Step {self.step_count} executed"


class ConcreteReActAgent(ReActAgent):
    name: str = "react_agent"
    description: str = "A test ReAct agent"
    think_result: bool = True
    act_result: str = "Action executed"

    async def think(self) -> bool:
        return self.think_result

    async def act(self) -> str:
        return self.act_result


class TestBaseAgent:
    def test_agent_creation(self):
        agent = ConcreteAgent()
        assert agent.name == "test_agent"
        assert agent.state == AgentState.IDLE
        assert agent.max_steps == 10
        assert agent.current_step == 0

    def test_agent_custom_settings(self):
        agent = ConcreteAgent(max_steps=5, duplicate_threshold=3)
        assert agent.max_steps == 5
        assert agent.duplicate_threshold == 3

    def test_agent_memory_initialization(self):
        agent = ConcreteAgent()
        assert isinstance(agent.memory, Memory)
        assert len(agent.memory.messages) == 0

    @pytest.mark.asyncio
    async def test_state_context(self):
        agent = ConcreteAgent()
        assert agent.state == AgentState.IDLE

        async with agent.state_context(AgentState.RUNNING):
            assert agent.state == AgentState.RUNNING

        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_state_context_error(self):
        agent = ConcreteAgent()

        with pytest.raises(ValueError):
            async with agent.state_context(AgentState.RUNNING):
                raise ValueError("Test error")

        assert agent.state == AgentState.IDLE

    def test_update_memory_user(self):
        agent = ConcreteAgent()
        agent.update_memory("user", "Hello")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "user"
        assert agent.memory.messages[0].content == "Hello"

    def test_update_memory_system(self):
        agent = ConcreteAgent()
        agent.update_memory("system", "System instruction")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "system"

    def test_update_memory_assistant(self):
        agent = ConcreteAgent()
        agent.update_memory("assistant", "Assistant response")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "assistant"

    def test_update_memory_tool(self):
        agent = ConcreteAgent()
        agent.update_memory("tool", "Tool result", name="test_tool", tool_call_id="123")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "tool"
        assert agent.memory.messages[0].name == "test_tool"

    def test_update_memory_invalid_role(self):
        agent = ConcreteAgent()
        with pytest.raises(ValueError):
            agent.update_memory("invalid", "content")

    def test_messages_property(self):
        agent = ConcreteAgent()
        agent.update_memory("user", "Hello")
        assert len(agent.messages) == 1
        assert agent.messages[0].content == "Hello"

    def test_messages_setter(self):
        agent = ConcreteAgent()
        new_messages = [Message.user_message("New message")]
        agent.messages = new_messages
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].content == "New message"

    def test_to_langchain_messages(self):
        agent = ConcreteAgent()
        agent.update_memory("user", "Hello")
        agent.update_memory("assistant", "Hi")
        lc_messages = agent.to_langchain_messages()
        assert len(lc_messages) == 2

    @pytest.mark.asyncio
    async def test_run(self):
        agent = ConcreteAgent(max_steps=5)
        result = await agent.run("Test request")
        assert "Step" in result

    @pytest.mark.asyncio
    async def test_run_with_max_steps(self):
        agent = ConcreteAgent(max_steps=2)
        result = await agent.run("Test request")
        assert "max steps" in result.lower() or "Step" in result

    @pytest.mark.asyncio
    async def test_run_from_non_idle_state(self):
        agent = ConcreteAgent()
        agent.state = AgentState.RUNNING
        with pytest.raises(RuntimeError):
            await agent.run("Test request")

    def test_is_stuck_false_initial(self):
        agent = ConcreteAgent()
        assert agent.is_stuck() is False

    def test_is_stuck_false_single_message(self):
        agent = ConcreteAgent()
        agent.update_memory("assistant", "Response")
        assert agent.is_stuck() is False

    def test_is_stuck_true(self):
        agent = ConcreteAgent(duplicate_threshold=2)
        agent.update_memory("assistant", "Same response")
        agent.update_memory("assistant", "Same response")
        agent.update_memory("assistant", "Same response")
        assert agent.is_stuck() is True

    def test_is_stuck_false_different_content(self):
        agent = ConcreteAgent()
        agent.update_memory("assistant", "Response 1")
        agent.update_memory("assistant", "Response 2")
        assert agent.is_stuck() is False

    def test_handle_stuck_state(self):
        agent = ConcreteAgent(next_step_prompt="Original prompt")
        agent.handle_stuck_state()
        assert "duplicate responses" in agent.next_step_prompt.lower()
        assert "Original prompt" in agent.next_step_prompt


class TestReActAgent:
    def test_react_agent_creation(self):
        agent = ConcreteReActAgent()
        assert agent.name == "react_agent"
        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_step_think_and_act(self):
        agent = ConcreteReActAgent()
        result = await agent.step()
        assert result == "Action executed"

    @pytest.mark.asyncio
    async def test_step_no_action_needed(self):
        agent = ConcreteReActAgent(think_result=False)
        result = await agent.step()
        assert "no action needed" in result.lower()

    @pytest.mark.asyncio
    async def test_react_run(self):
        agent = ConcreteReActAgent(max_steps=2)
        result = await agent.run("Test request")
        assert result is not None
