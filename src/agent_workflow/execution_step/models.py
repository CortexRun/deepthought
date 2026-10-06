from pydantic import BaseModel, Field
from enum import Enum


class Intent(str, Enum):
    OPERATING_SYSTEM = "operating_system"
    GENERAL = "general"


class IntentResult(BaseModel):
    intent: Intent = Field(
        ..., description="Whether the question is related to operating systems"
    )
