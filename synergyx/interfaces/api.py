"""FastAPI HTTP API for SynergyX."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ..config.manager import get_config, setup_logging
from ..core.engine import ChatEngine
from ..core.models import ChatRequest, ChatResponse, StreamingChatResponse
from ..tools.registry import get_registry, get_tool_schemas

logger = logging.getLogger(__name__)

# Initialize configuration and logging
config = get_config()
setup_logging(config)

# Create FastAPI app
app = FastAPI(
    title=config.get("api.title", "SynergyX API"),
    description=config.get("api.description", "Advanced AI Chatbot and Analysis System"),
    version=config.get("api.version", "0.1.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat engine
chat_engine = ChatEngine(config)


# Request/Response models
class ChatRequestAPI(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    use_tools: bool = True


class ChatResponseAPI(BaseModel):
    message: str
    conversation_id: str
    model: str
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    timestamp: datetime = datetime.now()


class AnalysisRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class AnalysisResponse(BaseModel):
    tool: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = datetime.now()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = datetime.now()
    providers: Dict[str, bool]
    tools: List[str]


# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        provider_status = await chat_engine.get_provider_status()
        registry = get_registry()
        tools = registry.list_tools()
        
        return HealthResponse(
            status="healthy",
            providers=provider_status,
            tools=tools
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/v1/chat", response_model=ChatResponseAPI)
async def chat_completion(request: ChatRequestAPI):
    """Chat completion endpoint."""
    try:
        # Get tools if requested
        tools = None
        if request.use_tools:
            tools = get_tool_schemas()
        
        # Create chat request
        chat_request = ChatRequest(
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Get response
        response = await chat_engine.chat(chat_request, tools=tools)
        
        return ChatResponseAPI(
            message=response.message,
            conversation_id=response.conversation_id,
            model=response.model,
            usage=response.usage,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason
        )
    
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/v1/chat/stream")
async def chat_completion_stream(request: ChatRequestAPI):
    """Streaming chat completion endpoint."""
    if not request.stream:
        request.stream = True
    
    try:
        # Get tools if requested
        tools = None
        if request.use_tools:
            tools = get_tool_schemas()
        
        # Create chat request
        chat_request = ChatRequest(
            message=request.message,
            conversation_id=request.conversation_id,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        async def generate_stream():
            try:
                async for chunk in chat_engine.stream_chat(chat_request, tools=tools):
                    # Format as SSE
                    chunk_data = {
                        "delta": chunk.delta,
                        "conversation_id": chunk.conversation_id,
                        "done": chunk.done,
                        "tool_calls": chunk.tool_calls,
                        "finish_reason": chunk.finish_reason
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send done signal
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_data = {"error": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Analysis tool endpoint."""
    try:
        registry = get_registry()
        result = await registry.execute_tool(request.tool_name, **request.parameters)
        
        return AnalysisResponse(
            tool=result["tool"],
            success=result["success"],
            result=result.get("result"),
            error=result.get("error")
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/tools")
async def list_tools():
    """List available analysis tools."""
    try:
        registry = get_registry()
        tools = registry.list_tools()
        schemas = registry.get_function_schemas()
        
        return {
            "tools": tools,
            "schemas": schemas
        }
    
    except Exception as e:
        logger.error(f"List tools failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/conversations")
async def list_conversations():
    """List all conversations."""
    try:
        conversations = await chat_engine.list_conversations()
        return {"conversations": conversations}
    
    except Exception as e:
        logger.error(f"List conversations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    try:
        conversation = await chat_engine.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {
            "id": conversation.id,
            "created_at": conversation.created_at.isoformat(),
            "metadata": conversation.metadata,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in conversation.messages
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        success = await chat_engine.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        return {"message": "Conversation deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/v1/providers")
async def get_providers():
    """Get provider status."""
    try:
        provider_status = await chat_engine.get_provider_status()
        return {"providers": provider_status}
    
    except Exception as e:
        logger.error(f"Get providers failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


if __name__ == "__main__":
    import uvicorn
    
    host = config.get_env("SYNERGYX_API_HOST", config.get("api.host", "0.0.0.0"))
    port = int(config.get_env("SYNERGYX_API_PORT", config.get("api.port", 8000)))
    
    uvicorn.run(app, host=host, port=port)