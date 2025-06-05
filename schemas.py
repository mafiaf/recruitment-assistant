from pydantic import BaseModel
from typing import List, Optional

class ResumeUpload(BaseModel):
    name: Optional[str] = ""
    text: str

class ChatRequest(BaseModel):
    text: str
    candidate_ids: List[str] = []
