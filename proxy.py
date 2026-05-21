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
NOVELAI_COMPLETIONS_URL = f"{NOVELAI_API_BASE}/oa/v1/completions"
NOVELAI_API_KEY = os.environ.get("NOVELAI_API_KEY", "")
DEFAULT_MODEL = "glm-4-6"
AVAILABLE_MODELS = os.environ.get("AVAILABLE_MODELS", "")

if not NOVELAI_API_KEY:
    raise RuntimeError(
        "Set NOVELAI_API_KEY in .env (copy env.example) or in your environment before starting the proxy"
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


def resolve_logprobs_count(body: dict[str, Any]) -> int:
    """Determine the integer logprobs count to request from NovelAI's completions endpoint.

    NovelAI's `/oa/v1/completions` endpoint (powered by vLLM) accepts ``logprobs`` as an
    integer N, returning top N+1 candidate tokens per position.

    Priority:
      1. ``top_logprobs: N`` (from client) → use N.
      2. ``logprobs: <int>`` (from client) → use the integer value.
      3. ``logprobs: true`` (Python bool) → default to ``5`` (a reasonable number).
      4. Otherwise → ``0`` (no logprobs requested).

    Note: ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is ``True``.  We must check for ``bool`` *before* ``int``.
    """
    top_logprobs = body.get("top_logprobs")
    if top_logprobs is not None:
        return max(1, int(top_logprobs))
    logprobs = body.get("logprobs")
    if isinstance(logprobs, bool):
        return 5 if logprobs else 0
    if isinstance(logprobs, int) and logprobs > 0:
        return logprobs
    return 0


def convert_completions_logprobs(
    comp_lp: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert completions-style logprobs to chat.completion.chunk style.

    Completions format (per chunk):
      ``{"tokens": [" to"], "token_logprobs": [-0.048], "top_logprobs": [{" to": -0.048}]}``

    Chat format (per chunk):
      ``{"content": [{"token": " to", "logprob": -0.048, "bytes": null, "top_logprobs": [...]}]}``
    """
    if not comp_lp:
        return None
    tokens = comp_lp.get("tokens") or []
    token_logprobs = comp_lp.get("token_logprobs") or []
    top_logprobs_list = comp_lp.get("top_logprobs") or []

    content: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        logprob = token_logprobs[i] if i < len(token_logprobs) else 0.0
        top_dict = top_logprobs_list[i] if i < len(top_logprobs_list) else {}

        # Build top_logprobs entries for this token position
        top_entries: list[dict[str, Any]] = []
        for t_token, t_logprob in top_dict.items():
            top_entries.append({
                "token": t_token,
                "logprob": t_logprob,
                "bytes": None,
            })

        content.append({
            "token": token,
            "logprob": logprob,
            "bytes": None,
            "top_logprobs": top_entries,
        })

    return {"content": content}


def build_novelai_body(body: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    payload = dict(body)
    # Remove unsupported params
    payload.pop("top_logprobs", None)
    # Strip boolean logprobs (chat endpoint rejects JSON booleans)
    if isinstance(payload.get("logprobs"), bool):
        payload.pop("logprobs", None)
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


def extract_stream_logprobs(chunk: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract per-token logprobs from a streaming chunk's first choice.

    Handles both formats:
    - Chat format: ``logprobs.content[{token, logprob, bytes, top_logprobs}]``
    - Completions format: ``logprobs.{tokens, token_logprobs, top_logprobs}``
    """
    choices = chunk.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    logprobs = choice.get("logprobs")
    if logprobs is None:
        return None

    # Chat format: content array with {token, logprob, ...}
    content = logprobs.get("content")
    if content:
        return content

    # Completions format: parallel arrays {tokens, token_logprobs, top_logprobs}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    top_logprobs_list = logprobs.get("top_logprobs") or []

    if not tokens:
        return None

    result: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        logprob = token_logprobs[i] if i < len(token_logprobs) else 0.0
        top_dict = top_logprobs_list[i] if i < len(top_logprobs_list) else {}
        top_entries: list[dict[str, Any]] = []
        for t_token, t_logprob in top_dict.items():
            top_entries.append({
                "token": t_token,
                "logprob": t_logprob,
                "bytes": None,
            })
        result.append({
            "token": token,
            "logprob": logprob,
            "bytes": None,
            "top_logprobs": top_entries,
        })

    return result


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

    # Convert completions-style logprobs to chat format when source is a
    # completions chunk (has "text" instead of "delta")
    raw_logprobs = choice.get("logprobs")
    chat_logprobs = convert_completions_logprobs(raw_logprobs) if raw_logprobs else raw_logprobs

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
                "logprobs": chat_logprobs,
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
    *,
    use_completions: bool = False,
) -> StreamingResponse:
    upstream = NOVELAI_COMPLETIONS_URL if use_completions else NOVELAI_CHAT_URL
    client, response = await open_novelai_stream(upstream, payload)

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
    *,
    use_completions: bool = False,
) -> tuple[str, str | None, str, dict[str, Any] | None, list[dict[str, Any]] | None]:
    upstream = NOVELAI_COMPLETIONS_URL if use_completions else NOVELAI_CHAT_URL
    completion_id: str | None = None
    model = payload.get("model", DEFAULT_MODEL)
    usage: dict[str, Any] | None = None
    full_text = ""
    logprobs: list[dict[str, Any]] | None = None

    async for event_type, chunk in iter_novelai_sse(client, upstream, payload):
        if event_type == "done":
            break
        assert chunk is not None
        completion_id = chunk.get("id") or completion_id
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        full_text += extract_stream_text(chunk)
        chunk_logprobs = extract_stream_logprobs(chunk)
        if chunk_logprobs is not None:
            if logprobs is None:
                logprobs = []
            logprobs.extend(chunk_logprobs)

    return full_text, completion_id, model, usage, logprobs


def openai_chat_response(
    content: str,
    completion_id: str | None,
    model: str,
    usage: dict[str, Any] | None,
    logprobs: list[dict[str, Any]] | None = None,
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
                "logprobs": {"content": logprobs} if logprobs else None,
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

    logprobs_count = resolve_logprobs_count(body)

    if logprobs_count > 0:
        # Use /oa/v1/completions endpoint which accepts logprobs as integer
        # (vLLM returns N+1 top candidates per token position).
        # Convert chat messages to a prompt for the completions endpoint.
        payload = dict(body)
        payload["model"] = requested_model
        payload["stream"] = True
        payload.pop("messages", None)
        payload.pop("top_logprobs", None)
        payload.pop("logprobs", None)
        payload["logprobs"] = logprobs_count
        if body.get("messages"):
            payload["prompt"] = messages_to_prompt(body["messages"])
        elif not payload.get("prompt"):
            raise HTTPException(status_code=400, detail="Request must include messages or prompt")
        # Ensure no leftover boolean logprobs in completions payload
        if isinstance(payload.get("logprobs"), bool):
            payload.pop("logprobs", None)
            payload["logprobs"] = max(1, logprobs_count)
        use_completions = True
    else:
        payload = build_novelai_body(body, stream=True)
        use_completions = False

    if client_stream:
        try:
            return await chat_streaming_response(payload, requested_model, use_completions=use_completions)
        except HTTPException:
            raise
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            full_text, completion_id, model, usage, logprobs = await stream_novelai_text(
                client, payload, use_completions=use_completions
            )
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"NovelAI request failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        openai_chat_response(full_text, completion_id, model or requested_model, usage, logprobs)
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
    # Remove unsupported params and resolve integer logprobs count
    payload.pop("top_logprobs", None)
    payload["logprobs"] = resolve_logprobs_count(body)
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
    logprobs: list[dict[str, Any]] | None = None

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
                chunk_logprobs = extract_stream_logprobs(chunk)
                if chunk_logprobs is not None:
                    if logprobs is None:
                        logprobs = []
                    logprobs.extend(chunk_logprobs)
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
                    "logprobs": {"content": logprobs} if logprobs else None,
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
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("proxy:app", host=host, port=port, reload=False)
