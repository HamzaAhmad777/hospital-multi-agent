from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Ye state poore graph mein har agent ke beech share hota hai.
    """
    messages: List[BaseMessage]      # conversation history (user + AI messages)
    patient_id: Optional[int]        # agar patient identify ho chuka hai
    intent: Optional[str]            # supervisor decide karega: appointment / history / general / emergency
    next_agent: Optional[str]        # konsa agent as next chalega
    is_emergency: bool               # emergency detect hui ya nahi