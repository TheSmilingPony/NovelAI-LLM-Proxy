"""
OpenAI-compatible proxy for NovelAI GLM 4.6.

NovelAI's non-streaming response uses completions-style `choices[].text` instead of
`choices[].message.content`, which front ends like SillyTavern reject.

- stream=false: request streaming from NovelAI, buffer, return chat.completion JSON.
- stream=true:  request streaming from NovelAI, passthrough as chat.completion.chunk SSE.
"""

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="NAIProxy")

NOVELAI_API_BASE = os.environ.get("NOVELAI_API_BASE", "https://text.novelai.net").rstrip("/")
NOVELAI_CHAT_URL = f"{NOVELAI_API_BASE}/oa/v1/chat/completions"
NOVELAI_API_KEY = os.environ.get("NOVELAI_API_KEY", "")
DEFAULT_MODEL = "glm-4-6"
AVAILABLE_MODELS = os.environ.get("AVAILABLE_MODELS", "")

if not NOVELAI_API_KEY:
    raise RuntimeError(
        "Set NOVELAI_API_KEY in .env (copy .env.example) or in your environment before starting the proxy"
    )


def normalize_model(model: str | None) -> str:
    if not model:
        return DEFAULT_MODEL
    return model.replace("glm-4.6", "glm-4-6").replace("GLM-4.6", "glm-4-6")


def message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Fallback when upstream only accepts a prompt string."""
    blocks: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = message_content(msg.get("content"))
        if not text:
            continue
        if role == "system":
            blocks.append(text)
        elif role == "assistant":
            blocks.append(f"Assistant: {text}")
        else:
            blocks.append(text)
    return "\n\n".join(blocks)


def novelai_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {NOVELAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def build_novelai_body(body: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    payload = dict(body)
    payload["model"] = normalize_model(payload.get("model"))
    payload["stream"] = stream

    # SillyTavern sends chat messages; NovelAI chat endpoint accepts them directly.
    if not payload.get("messages") and not payload.get("prompt"):
        raise HTTPException(status_code=400, detail="Request must include messages or prompt")

    return payload


def extract_stream_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]

    delta = choice.get("delta") or {}
    content = delta.get("content")
    if content:
        return content

    text = choice.get("text")
    if text:
        return text

    message = choice.get("message") or {}
    return message.get("content") or ""


def normalize_chat_stream_chunk(chunk: dict[str, Any], fallback_model: str) -> dict[str, Any] | None:
    """Ensure upstream SSE chunks match OpenAI chat.completion.chunk shape."""
    choices = chunk.get("choices")
    if not choices:
        return None

    choice = choices[0]
    completion_id = chunk.get("id") or f"chatcmpl-{uuid.uuid4()}"
    created = chunk.get("created", int(time.time()))
    model = chunk.get("model") or fallback_model

    if "delta" in choice:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "delta": choice.get("delta") or {},
                    "finish_reason": choice.get("finish_reason"),
                    "logprobs": choice.get("logprobs"),
                }
            ],
            **({"usage": chunk["usage"]} if chunk.get("usage") else {}),
        }

    text = choice.get("text")
    finish_reason = choice.get("finish_reason")
    if not text and finish_reason is None:
        return None

    delta: dict[str, Any] = {}
    if text:
        delta["content"] = text

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": choice.get("index", 0),
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": choice.get("logprobs"),
            }
        ],
        **({"usage": chunk["usage"]} if chunk.get("usage") else {}),
    }


async def iter_novelai_sse(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
    """Yield (event_type, parsed_chunk). event_type is 'data' or 'done'."""
    async with client.stream(
        "POST",
        url,
        headers=novelai_headers(),
        json=payload,
    ) as response:
        if response.status_code >= 400:
            detail = await response.aread()
            raise HTTPException(
                status_code=response.status_code,
                detail=detail.decode("utf-8", errors="replace"),
            )

        async for raw_line in response.aiter_lines():
            if not raw_line or raw_line.startswith(":"):
                continue
            if not raw_line.startswith("data:"):
                continue

            data = raw_line[5:].lstrip()
            if data == "[DONE]":
                yield ("done", None)
                return

            try:
                yield ("data", json.loads(data))
            except json.JSONDecodeError:
                continue


async def relay_sse(
    response: httpx.Response,
    fallback_model: str,
    *,
    normalize_chat: bool,
) -> AsyncIterator[bytes]:
    try:
        async for raw_line in response.aiter_lines():
            if not raw_line or raw_line.startswith(":"):
                continue
            if not raw_line.startswith("data:"):
                continue

            data = raw_line[5:].lstrip()
            if data == "[DONE]":
                yield b"data: [DONE]\n\n"
                return

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if normalize_chat:
                normalized = normalize_chat_stream_chunk(chunk, fallback_model)
                if not normalized:
                    continue
                chunk = normalized

            line = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {line}\n\n".encode()
    finally:
        await response.aclose()

    yield b"data: [DONE]\n\n"


async def open_novelai_stream(
    url: str,
    payload: dict[str, Any],
) -> tuple[httpx.AsyncClient, httpx.Response]:
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    request = client.build_request(
        "POST",
        url,
        headers=novelai_headers(),
        json=payload,
    )
    response = await client.send(request, stream=True)
    if response.status_code >= 400:
        detail = await response.aread()
        await response.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=response.status_code,
            detail=detail.decode("utf-8", errors="replace"),
        )
    return client, response


async def chat_streaming_response(
    payload: dict[str, Any],
    fallback_model: str,
) -> StreamingResponse:
    client, response = await open_novelai_stream(NOVELAI_CHAT_URL, payload)

    async def cleanup_wrapper() -> AsyncIterator[bytes]:
        try:
            async for part in relay_sse(response, fallback_model, normalize_chat=True):
                yield part
        finally:
            await client.aclose()

    return StreamingResponse(
        cleanup_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def stream_novelai_text(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> tuple[str, str | None, str, dict[str, Any] | None]:
    completion_id: str | None = None
    model = payload.get("model", DEFAULT_MODEL)
    usage: dict[str, Any] | None = None
    full_text = ""

    async for event_type, chunk in iter_novelai_sse(client, NOVELAI_CHAT_URL, payload):
        if event_type == "done":
            break
        assert chunk is not None
        completion_id = chunk.get("id") or completion_id
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        full_text += extract_stream_text(chunk)

    return full_text, completion_id, model, usage


def openai_chat_response(
    content: str,
    completion_id: str | None,
    model: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": completion_id or f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def list_models():
    if AVAILABLE_MODELS:
        models = [m.strip() for m in AVAILABLE_MODELS.split(",") if m.strip()]
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "owned_by": "novelai",
                }
                for m in models
            ],
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = novelai_headers()
            # The default novelai_headers asks for SSE streams; we just want JSON here.
            headers["Accept"] = "application/json" 
            
            response = await client.get(
                f"{NOVELAI_API_BASE}/oa/v1/models",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                # If NovelAI returns a valid OpenAI-style list, pass it to SillyTavern
                if "data" in data:
                    return data
    except Exception as e:
        print(f"Warning: Failed to dynamically fetch models from NovelAI - {e}")

    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "owned_by": "novelai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested_model = normalize_model(body.get("model"))
    client_stream = bool(body.get("stream"))

    payload = build_novelai_body(body, stream=True)

    if client_stream:
        try:
            return await chat_streaming_response(payload, requested_model)
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            full_text, completion_id, model, usage = await stream_novelai_text(client, payload)
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        openai_chat_response(full_text, completion_id, model or requested_model, usage)
    )


async def completions_streaming_response(payload: dict[str, Any]) -> StreamingResponse:
    upstream_url = f"{NOVELAI_API_BASE}/oa/v1/completions"
    client, response = await open_novelai_stream(upstream_url, payload)

    async def cleanup_wrapper() -> AsyncIterator[bytes]:
        try:
            async for part in relay_sse(response, DEFAULT_MODEL, normalize_chat=False):
                yield part
        finally:
            await client.aclose()

    return StreamingResponse(
        cleanup_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/completions")
async def completions(request: Request):
    """Optional passthrough for callers that hit /v1/completions instead of chat."""
    body = await request.json()
    requested_model = normalize_model(body.get("model"))
    client_stream = bool(body.get("stream"))

    payload = dict(body)
    payload["model"] = requested_model
    payload["stream"] = True

    if payload.get("messages") and not payload.get("prompt"):
        payload["prompt"] = messages_to_prompt(payload["messages"])
        del payload["messages"]

    if not payload.get("prompt") and not payload.get("messages"):
        raise HTTPException(status_code=400, detail="Request must include prompt or messages")

    upstream_url = f"{NOVELAI_API_BASE}/oa/v1/completions"

    if client_stream:
        try:
            return await completions_streaming_response(payload)
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc

    full_text = ""
    completion_id: str | None = None
    usage: dict[str, Any] | None = None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            async for event_type, chunk in iter_novelai_sse(client, upstream_url, payload):
                if event_type == "done":
                    break
                assert chunk is not None
                completion_id = chunk.get("id") or completion_id
                if chunk.get("usage"):
                    usage = chunk["usage"]
                full_text += extract_stream_text(chunk)
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc

    return JSONResponse(
        {
            "id": completion_id or f"cmpl-{uuid.uuid4()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "text": full_text,
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": usage
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("proxy:app", host=host, port=port, reload=False)
