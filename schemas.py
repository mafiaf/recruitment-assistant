from pydantic import BaseModel
from typing import List, Optional

class ResumeUpload(BaseModel):
    name: Optional[str] = ""
    text: str
    skills: Optional[List[str]] = None
    location: Optional[str] = None
    years: Optional[int] = None
    tags: Optional[List[str]] = None

class ChatRequest(BaseModel):
    text: str
    candidate_ids: List[str] = []
