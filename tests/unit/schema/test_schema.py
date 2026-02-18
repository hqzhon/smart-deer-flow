from src.schema import (
    Role,
    ToolChoice,
    AgentState,
    Function,
    ToolCall,
    Message,
    Memory,
)


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_role_enum_membership(self):
        assert Role("system") == Role.SYSTEM
        assert Role("user") == Role.USER


class TestToolChoice:
    def test_tool_choice_values(self):
        assert ToolChoice.NONE == "none"
        assert ToolChoice.AUTO == "auto"
        assert ToolChoice.REQUIRED == "required"


class TestAgentState:
    def test_agent_state_values(self):
        assert AgentState.IDLE == "IDLE"
        assert AgentState.RUNNING == "RUNNING"
        assert AgentState.FINISHED == "FINISHED"
        assert AgentState.ERROR == "ERROR"


class TestFunction:
    def test_function_creation(self):
        func = Function(name="test_func", arguments='{"arg1": "value1"}')
        assert func.name == "test_func"
        assert func.arguments == '{"arg1": "value1"}'


class TestToolCall:
    def test_tool_call_creation(self):
        func = Function(name="test_func", arguments="{}")
        tool_call = ToolCall(id="call_123", type="function", function=func)
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "test_func"

    def test_tool_call_default_type(self):
        func = Function(name="test_func", arguments="{}")
        tool_call = ToolCall(id="call_123", function=func)
        assert tool_call.type == "function"


class TestMessage:
    def test_user_message(self):
        msg = Message.user_message("Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"
        assert msg.base64_image is None

    def test_user_message_with_image(self):
        msg = Message.user_message("Hello", base64_image="base64string")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.base64_image == "base64string"

    def test_system_message(self):
        msg = Message.system_message("System instruction")
        assert msg.role == Role.SYSTEM
        assert msg.content == "System instruction"

    def test_assistant_message(self):
        msg = Message.assistant_message("Assistant response")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Assistant response"

    def test_assistant_message_empty(self):
        msg = Message.assistant_message()
        assert msg.role == Role.ASSISTANT
        assert msg.content is None

    def test_tool_message(self):
        msg = Message.tool_message(
            "Tool result", name="test_tool", tool_call_id="call_123"
        )
        assert msg.role == Role.TOOL
        assert msg.content == "Tool result"
        assert msg.name == "test_tool"
        assert msg.tool_call_id == "call_123"

    def test_from_tool_calls(self):
        func = Function(name="test_func", arguments="{}")
        tool_call = ToolCall(id="call_123", function=func)
        msg = Message.from_tool_calls([tool_call], content="Thinking...")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Thinking..."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"

    def test_to_dict(self):
        msg = Message.user_message("Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_dict_with_tool_calls(self):
        func = Function(name="test_func", arguments="{}")
        tool_call = ToolCall(id="call_123", function=func)
        msg = Message.from_tool_calls([tool_call])
        d = msg.to_dict()
        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1

    def test_add_message_to_list(self):
        msg = Message.user_message("Hello")
        result = msg + [Message.system_message("System")]
        assert len(result) == 2
        assert result[0].role == Role.USER
        assert result[1].role == Role.SYSTEM

    def test_add_two_messages(self):
        msg1 = Message.user_message("Hello")
        msg2 = Message.assistant_message("Hi")
        result = msg1 + msg2
        assert len(result) == 2

    def test_radd_message(self):
        msg = Message.user_message("Hello")
        result = [Message.system_message("System")] + msg
        assert len(result) == 2

    def test_to_langchain_user(self):
        msg = Message.user_message("Hello")
        lc_msg = msg.to_langchain()
        assert lc_msg.content == "Hello"
        assert lc_msg.type == "human"

    def test_to_langchain_system(self):
        msg = Message.system_message("System instruction")
        lc_msg = msg.to_langchain()
        assert lc_msg.content == "System instruction"
        assert lc_msg.type == "system"

    def test_to_langchain_assistant(self):
        msg = Message.assistant_message("Response")
        lc_msg = msg.to_langchain()
        assert lc_msg.content == "Response"
        assert lc_msg.type == "ai"

    def test_to_langchain_tool(self):
        msg = Message.tool_message("Result", name="tool", tool_call_id="123")
        lc_msg = msg.to_langchain()
        assert lc_msg.content == "Result"
        assert lc_msg.tool_call_id == "123"


class TestMemory:
    def test_memory_creation(self):
        memory = Memory()
        assert len(memory.messages) == 0
        assert memory.max_messages == 100

    def test_memory_custom_max(self):
        memory = Memory(max_messages=50)
        assert memory.max_messages == 50

    def test_add_message(self):
        memory = Memory()
        msg = Message.user_message("Hello")
        memory.add_message(msg)
        assert len(memory.messages) == 1
        assert memory.messages[0].content == "Hello"

    def test_add_messages(self):
        memory = Memory()
        messages = [Message.user_message("Hello"), Message.assistant_message("Hi")]
        memory.add_messages(messages)
        assert len(memory.messages) == 2

    def test_max_messages_limit(self):
        memory = Memory(max_messages=3)
        for i in range(5):
            memory.add_message(Message.user_message(f"Message {i}"))
        assert len(memory.messages) == 3
        assert memory.messages[0].content == "Message 2"
        assert memory.messages[-1].content == "Message 4"

    def test_clear(self):
        memory = Memory()
        memory.add_message(Message.user_message("Hello"))
        memory.clear()
        assert len(memory.messages) == 0

    def test_get_recent_messages(self):
        memory = Memory()
        for i in range(5):
            memory.add_message(Message.user_message(f"Message {i}"))
        recent = memory.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[-1].content == "Message 4"

    def test_to_dict_list(self):
        memory = Memory()
        memory.add_message(Message.user_message("Hello"))
        memory.add_message(Message.assistant_message("Hi"))
        dict_list = memory.to_dict_list()
        assert len(dict_list) == 2
        assert dict_list[0]["role"] == "user"
        assert dict_list[1]["role"] == "assistant"
