from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator
from app.districts import DistrictId

Language = Literal["ar", "en"]
Category = Literal["lighting", "roads", "waste", "water", "parking", "noise", "other"]
Onset = Literal["today", "week", "month", "unknown"]


class ComplaintCreate(BaseModel):
    language: Language = "ar"
    full_name: str = Field(min_length=2, max_length=160)
    phone: str = Field(min_length=7, max_length=40)
    email: Optional[EmailStr] = None
    category: Category
    district: DistrictId
    address: str = Field(min_length=2, max_length=300)
    description: str = Field(min_length=8, max_length=8000)
    onset: Onset
    consent: bool
    attachment_name: Optional[str] = Field(default=None, max_length=255)

    @field_validator("full_name", "phone", "district", "address", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class ComplaintCreated(BaseModel):
    reference_number: str
    status: str
    created_at: datetime
    message: str


class ComplaintLookup(BaseModel):
    reference_number: str
    phone: str = Field(min_length=7, max_length=40)


class ComplaintStatus(BaseModel):
    reference_number: str
    category: Category
    district: DistrictId
    status: str
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    language: Language = "ar"
    message: str = Field(min_length=1, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str
    provider: str = "cohere"
    model: str
    ready_for_review: bool = False
    summary: Optional[str] = None
    category: Optional[Category] = None
    district: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    onset: Optional[Onset] = None

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SalesAnalystChatRequest(BaseModel):
    language: Literal["ar", "en"] = "en"
    message: str = Field(min_length=2, max_length=4000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=12)
    dashboard_context: Dict[str, Any] = Field(default_factory=dict)


class SalesAnalystChatResponse(BaseModel):
    reply: str
    provider: str = "cohere"
    model: str
