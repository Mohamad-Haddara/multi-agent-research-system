"""
API Schemas (Pydantic)

Contracts

This file contains chat schema for the application.

The Request and Response models of the public API.

Our models are API schemas, not internal types.
The are contract with outside world over HTTP. Pydantic here checks what the clients sends, and turns the response into a JSON object.

The conversion happens in the route
"""

from time import timezone
from typing import Literal, List, Annotated
from uuid import uuid4, UUID


from pydantic import (
    BaseModel,
    Field,
    field_validator,
    IPvAnyAddress,
    StringConstraints
)



from datetime import datetime, timezone
from src.schemas.base import BaseResponse

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint. (Body of POST /chat)

    Attributes:
        messages: User input prompt
        thread_id: Optional field to any valid UUID
    """
    messages: Annotated[str, StringConstraints(min_length=1, max_length=4000,strip_whitespace=True)]
    thread_id: UUID | None = Field(default=None, description="Omit to start a new conversation")


class ChatResponse(BaseResponse):
    """
    Response model for chat endpoint. (Reply from POST /chat)

    Attributes:
        request_id: Generate a new request UUID by passing callable to default_factory that will be called to generate new UUID
        ip: Optional field to any valid IPv4 or IPv6 address ranges
        content: List of messages in the conversation
        created_at: Time the message is respond
    """

    #request_id: Annotated[str, Field(default_factory= lambda:uuid4().hex)]
    request_id: str | UUID
    thread_id: str | UUID = Field(description="The conversation this reply belongs to")
    #ip: Annotated[str, IPvAnyAddress]
    content: str = Field(description="The assistant's reply")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StreamResponse(BaseModel):
    """
    Response model for chat endpoint.

    Attributes:
        content: The content of the current chunk
        done: Whether the stream is complete 
    """
    messages: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default="", description="Whether the stream is complete")


