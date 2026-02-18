# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/browser", tags=["browser"])

_browser_sessions: Dict[str, "BrowserSession"] = {}


class BrowserAction(BaseModel):
    action: str = Field(..., description="Browser action to perform")
    url: Optional[str] = Field(None, description="URL for navigation actions")
    index: Optional[int] = Field(None, description="Element index")
    text: Optional[str] = Field(None, description="Text for input/selection")
    scroll_amount: Optional[int] = Field(None, description="Scroll amount in pixels")
    tab_id: Optional[int] = Field(None, description="Tab ID for switching")
    goal: Optional[str] = Field(None, description="Extraction goal")
    keys: Optional[str] = Field(None, description="Keys to send")
    seconds: Optional[int] = Field(None, description="Wait duration in seconds")
    selector: Optional[str] = Field(None, description="CSS selector")


class BrowserActionRequest(BaseModel):
    action: BrowserAction = Field(..., description="Browser action to execute")


class BrowserActionRecord(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    params: Dict[str, Any] = {}
    result: str
    success: bool = True


class BrowserState(BaseModel):
    session_id: str
    url: str = ""
    title: str = ""
    tabs: List[Dict[str, Any]] = []
    interactive_elements: List[Dict[str, Any]] = []
    last_screenshot: Optional[str] = None
    actions: List[BrowserActionRecord] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BrowserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tool = None
        self.state = BrowserState(session_id=session_id)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        if self._initialized:
            return

        from src.tools.browser.browser_use import BrowserUseTool

        self.tool = BrowserUseTool()
        self._initialized = True

    async def execute_action(self, action: BrowserAction) -> Dict[str, Any]:
        async with self._lock:
            if not self._initialized:
                await self.initialize()

            result = await self.tool.async_execute(
                action=action.action,
                url=action.url,
                index=action.index,
                text=action.text,
                scroll_amount=action.scroll_amount,
                tab_id=action.tab_id,
                goal=action.goal,
                keys=action.keys,
                seconds=action.seconds,
                selector=action.selector,
            )

            record = BrowserActionRecord(
                action=action.action,
                params=action.model_dump(exclude_none=True),
                result=result.output or result.error or "",
                success=result.error is None,
            )
            self.state.actions.append(record)
            self.state.updated_at = datetime.utcnow()

            if result.base64_image:
                self.state.last_screenshot = result.base64_image

            if action.action == "get_state":
                try:
                    state_data = json.loads(result.output or "{}")
                    self.state.url = state_data.get("url", "")
                    self.state.title = state_data.get("title", "")
                    self.state.tabs = state_data.get("tabs", [])
                    self.state.interactive_elements = state_data.get(
                        "interactive_elements", []
                    )
                except json.JSONDecodeError:
                    pass

            return {
                "output": result.output,
                "error": result.error,
                "base64_image": result.base64_image,
            }

    async def get_screenshot(self) -> Optional[str]:
        if not self._initialized:
            await self.initialize()

        result = await self.tool.async_execute(action="screenshot")
        if result.base64_image:
            self.state.last_screenshot = result.base64_image
            return result.base64_image
        return None

    async def get_state(self) -> BrowserState:
        if not self._initialized:
            return self.state

        result = await self.tool.async_execute(action="get_state")
        try:
            state_data = json.loads(result.output or "{}")
            self.state.url = state_data.get("url", "")
            self.state.title = state_data.get("title", "")
            self.state.tabs = state_data.get("tabs", [])
            self.state.interactive_elements = state_data.get("interactive_elements", [])
        except json.JSONDecodeError:
            pass

        return self.state

    async def cleanup(self):
        if self.tool:
            await self.tool.cleanup()
        self._initialized = False


def get_or_create_session(session_id: str) -> BrowserSession:
    if session_id not in _browser_sessions:
        _browser_sessions[session_id] = BrowserSession(session_id)
    return _browser_sessions[session_id]


@router.post("/{session_id}/execute")
async def execute_browser_action(session_id: str, request: BrowserActionRequest):
    """Execute a browser action."""
    session = get_or_create_session(session_id)

    try:
        result = await session.execute_action(request.action)
        return {"success": True, "session_id": session_id, "result": result}
    except Exception as e:
        logger.error(f"Browser action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/screenshot")
async def get_screenshot(session_id: str):
    """Get a screenshot of the current browser state."""
    session = get_or_create_session(session_id)

    try:
        screenshot = await session.get_screenshot()
        if screenshot:
            return {
                "success": True,
                "session_id": session_id,
                "base64_image": screenshot,
                "url": session.state.url,
            }
        return {"success": False, "error": "Failed to capture screenshot"}
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/state", response_model=BrowserState)
async def get_browser_state(session_id: str):
    """Get the current browser state."""
    session = get_or_create_session(session_id)

    try:
        state = await session.get_state()
        return state
    except Exception as e:
        logger.error(f"Get state failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/actions")
async def get_browser_actions(
    session_id: str,
    limit: int = Query(
        50, ge=1, le=500, description="Maximum number of actions to return"
    ),
):
    """Get the action history for a browser session."""
    session = get_or_create_session(session_id)
    actions = session.state.actions[-limit:]
    return {
        "session_id": session_id,
        "actions": actions,
        "total": len(session.state.actions),
    }


@router.delete("/{session_id}")
async def close_browser_session(session_id: str):
    """Close and cleanup a browser session."""
    if session_id not in _browser_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _browser_sessions[session_id]
    await session.cleanup()
    del _browser_sessions[session_id]

    return {"success": True, "message": f"Session {session_id} closed"}


@router.get("")
async def list_browser_sessions():
    """List all active browser sessions."""
    sessions = []
    for session_id, session in _browser_sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "url": session.state.url,
                "title": session.state.title,
                "tabs_count": len(session.state.tabs),
                "actions_count": len(session.state.actions),
                "created_at": session.state.created_at,
                "updated_at": session.state.updated_at,
            }
        )
    return {"sessions": sessions, "total": len(sessions)}


@router.websocket("/{session_id}/stream")
async def browser_screenshot_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time screenshot streaming."""
    await websocket.accept()

    session = get_or_create_session(session_id)
    streaming = True
    interval = 1.0

    try:
        while streaming:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if data.get("type") == "stop":
                    streaming = False
                    break
                elif data.get("type") == "set_interval":
                    interval = max(0.5, min(5.0, data.get("interval", 1.0)))
                elif data.get("type") == "action":
                    action = BrowserAction(**data.get("action", {}))
                    result = await session.execute_action(action)
                    await websocket.send_json(
                        {"type": "action_result", "result": result}
                    )
            except asyncio.TimeoutError:
                pass

            screenshot = await session.get_screenshot()
            if screenshot:
                await websocket.send_json(
                    {
                        "type": "screenshot",
                        "base64_image": screenshot,
                        "url": session.state.url,
                        "title": session.state.title,
                    }
                )

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
