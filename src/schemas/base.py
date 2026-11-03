"""
Base response schemas shared across all the endpoints.
"""

from uuid import UUID, uuid4

from asgi_correlation_id import correlation_id # is a Python middleware package for ASGI frameworks (such as FastAPI, Starlette, Uvicorn)
# that reads or generates unique request correlation IDs to trace logs across asynchronous requests
# Automatic ID Generation/Extraction: Checks incoming HTTP or WebSocket requests for an existing header (default X-Request-ID or X-Correlation-ID) or generates a new UUIDv4 if none is found

from pydantic import BaseModel, Field



def _get_request_id() -> UUID:
    """
    Return the current request's correlation ID, or a fresh UUID as fallback.
    """

    value = correlation_id.get()
    print(f"Value: {value}" )

    if value is None:
        raise RuntimeError("CorrelationIdMiddlewate is not installed")

    return UUID(value)



class BaseResponse(BaseModel):
    """
    Base response model that all endpoint responses inherit from.

    request_id is auto-populated from the CorrelationIdMiddleware ContextVar - 
    no endpoint needs to pass it explicitly

    """

    request_id: UUID = Field(default_factory=_get_request_id, description="Unique identifier for this request")


