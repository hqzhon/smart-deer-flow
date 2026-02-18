from abc import abstractmethod

from src.agents.base import BaseAgent


class ReActAgent(BaseAgent):
    @abstractmethod
    async def think(self) -> bool:
        pass

    @abstractmethod
    async def act(self) -> str:
        pass

    async def step(self) -> str:
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
