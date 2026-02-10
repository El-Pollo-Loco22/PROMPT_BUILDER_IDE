from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PromptBlock(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    metadata: Optional[dict] = None


class PromptDocument(BaseModel):
    blocks: list[PromptBlock]
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
