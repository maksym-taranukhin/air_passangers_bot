from typing import Dict, List, Optional
from typing_extensions import TypedDict


class ChatRequest(TypedDict):
    question: str
    chat_history: Optional[List[Dict[str, str]]]
