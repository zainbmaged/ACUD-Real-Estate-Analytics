from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    SalesAnalystChatRequest,
    SalesAnalystChatResponse,
)
from app.services.cohere_chat import generate_reply  # Existing complaints intake
from app.services.sales_analyst_chat import generate_sales_reply

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
executor = ThreadPoolExecutor(max_workers=4)


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """Existing complaint-intake assistant. Kept backward-compatible."""
    settings = get_settings()
    if not settings.cohere_api_key:
        raise HTTPException(
            status_code=503,
            detail="The chat assistant is not configured yet. Add COHERE_API_KEY to backend/.env.",
        )
    future = executor.submit(generate_reply, payload.language, payload.message, payload.history)
    try:
        result = future.result(timeout=35)
    except FutureTimeout:
        future.cancel()
        raise HTTPException(status_code=504, detail="The assistant took too long to respond. Please try again.")
    except Exception as exc:
        print(f"Cohere complaint chat error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="The assistant is temporarily unavailable. Please try again.")
    return ChatResponse(reply=result["reply"], model=settings.cohere_model, **{
        key: value for key, value in result.items() if key != "reply"
    })


@router.post("/sales", response_model=SalesAnalystChatResponse)
def sales_analyst_chat(payload: SalesAnalystChatRequest):
    """Dashboard-grounded sales analyst; separate from complaint intake."""
    settings = get_settings()
    if not settings.cohere_api_key:
        raise HTTPException(
            status_code=503,
            detail="The sales analyst is not configured. Add COHERE_API_KEY to backend/.env.",
        )
    future = executor.submit(
        generate_sales_reply,
        payload.language,
        payload.message,
        payload.history,
        payload.dashboard_context,
    )
    try:
        reply = future.result(timeout=35)
    except FutureTimeout:
        future.cancel()
        raise HTTPException(status_code=504, detail="The analyst took too long to respond. Please try again.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        print(f"Cohere sales analyst error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="The sales analyst is temporarily unavailable. Please try again.")
    return SalesAnalystChatResponse(reply=reply, model=settings.cohere_model)
