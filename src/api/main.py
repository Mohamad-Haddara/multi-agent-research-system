"""
This file contains the main application entry point. ASGI entrypoint.

FastAPI; lifespan builds
"""

import logging
from uuid import uuid4


from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.routing import JSONResponse

from src.core.config import get_settings
from datetime import datetime

from openai import OpenAI
from src.core.config import get_settings

from src.schemas.chat import ChatRequest, ChatResponse



settings = get_settings()



logger = logging.getLogger(__name__)

client = OpenAI(api_key = settings.openai_api_key)

app = FastAPI(
   # title= settings.project_name,
    version= settings.version,
    description= settings.description,


)


@app.get("/")
async def root(request: Request) -> dict[str, str]:
    """Root endpoint returning basic API information"""
    logger.info("root endpoint called")
    return {
        "name": settings.project_name,
        "version": settings.version,
        "status": "healthy",
        "environment": settings.app_env,
        "swagger_url": "/doc"
    }




@app.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """
    Health check endpoint with environment-specific information.

    Here I should check if the dependencies (database) are reachable

    Returns:
        JSONResponse: Health status payload, with HTTP 503 when the database is 
        unreachable so load balancer can drop the instance
    """

    logger.info("health_check endpoint called")
    
    response = {
        "status": "healthy",
        "version": settings.version,
        "environment": settings.app_env,
        "timestamp": datetime.now().isoformat(),
    }

    return JSONResponse(content=response)




# API Routes
@app.post("/chat")
async def chat(body: ChatRequest, request: Request ) -> ChatResponse:
    """
    Chat endpoint using OpenAI API
    """
    logger.info("Chat endpoint called")
    
    messages = [
        {
            "role": "system",
            "content" : " You are a helpful assistant that can answer questions and help with tasks."
        },
        {
            "role":"user",
            "content": body.messages
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        logprobs=None,
        logit_bias=None,
        user=None,
        response_format=None,
    )

    # Pydantic validates the value is UUID and converts it to a string in the JSON output. Recommendation use UUID in both request and response model
    return ChatResponse(
        request_id= uuid4(),
        thread_id=body.thread_id or uuid4(), 
        content = response.choices[0].message.content
    )


if __name__ == "__main__":
    import uvicorn
    # Use reload in development only. In production, remove it.
    # Open the auto-generated API docs at http://localhost:8000/docs.
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)