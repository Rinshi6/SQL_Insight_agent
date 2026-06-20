from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from main import graph_sql


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _configure_backend_logging() -> tuple[logging.Logger, Path]:
    log_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path("logs") / log_date
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    info_handler = logging.FileHandler(log_dir / "backend.log", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(log_dir / "backend_error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    loggers = [
        logging.getLogger(),
        logging.getLogger("uvicorn"),
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
        logging.getLogger("txt2sql.backend"),
    ]

    for logger_item in loggers:
        logger_item.setLevel(logging.INFO)
        existing_files = {
            getattr(handler, "baseFilename", None)
            for handler in logger_item.handlers
        }
        for handler in (info_handler, error_handler):
            if handler.baseFilename not in existing_files:
                logger_item.addHandler(handler)

    stdout_file = open(log_dir / "backend_stdout.log", "a", encoding="utf-8", buffering=1)
    stderr_file = open(log_dir / "backend_stderr.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, stdout_file)
    sys.stderr = _TeeStream(sys.__stderr__, stderr_file)

    logger = logging.getLogger("txt2sql.backend")
    logger.info("Backend logging initialized in %s", log_dir)
    return logger, log_dir


logger, LOG_DIR = _configure_backend_logging()

app = FastAPI(title="Text-to-SQL Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, list[dict[str, Any]]] = {}
_session_lock = RLock()


@app.middleware("http")
async def log_requests(request, call_next):
    started_at = time.perf_counter()
    logger.info("request_started method=%s path=%s", request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise

    duration = time.perf_counter() - started_at
    logger.info(
        "request_finished method=%s path=%s status_code=%s duration_seconds=%.4f",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sql: str | None = None
    generated_sql: str | None = None
    explanation: str | None = None
    is_valid: bool | str | None = None
    history: list[ChatMessage]
    raw_result: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _get_field(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _build_contextual_query(session_history: list[dict[str, Any]], message: str) -> str:
    recent_turns = session_history[-6:]
    if not recent_turns:
        return message

    context_lines = [
        f"{item['role']}: {item['content']}"
        for item in recent_turns
        if item.get("content")
    ]
    context = "\n".join(context_lines)
    return (
        "Use the previous chat only if the new user question depends on it.\n\n"
        f"Previous chat:\n{context}\n\n"
        f"Current user question:\n{message}"
    )


def _format_answer(result: dict[str, Any]) -> tuple[str, str | None, str | None, str | None, bool | str | None]:
    sql_generation = result.get("sql_query")
    validation = result.get("final_query")

    generated_sql = _get_field(sql_generation, "sql_query")
    final_sql = _get_field(validation, "final_sql") or generated_sql
    explanation = _get_field(validation, "explanation")
    is_valid = _get_field(validation, "is_valid")

    if final_sql:
        answer = f"Here is the validated SQL query:\n\n```sql\n{final_sql}\n```"
        if explanation:
            answer += f"\n\nValidation note: {explanation}"
    else:
        answer = "The agent finished, but it did not return a SQL query."

    return answer, final_sql, generated_sql, explanation, is_valid


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "log_dir": str(LOG_DIR)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or uuid4().hex
    message = request.message.strip()
    logger.info("chat_received session_id=%s message_length=%s", session_id, len(message))

    with _session_lock:
        history = _sessions.setdefault(session_id, [])
        contextual_query = _build_contextual_query(history, message)
        history.append({"role": "user", "content": message, "created_at": _now(), "metadata": {}})

    try:
        result = await run_in_threadpool(graph_sql, {"user_query": contextual_query})
    except Exception as exc:
        logger.exception("chat_failed session_id=%s", session_id)
        with _session_lock:
            history = _sessions.setdefault(session_id, [])
            history.append(
                {
                    "role": "assistant",
                    "content": f"Agent error: {exc}",
                    "created_at": _now(),
                    "metadata": {"error": repr(exc)},
                }
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    safe_result = _jsonable(result)
    answer, final_sql, generated_sql, explanation, is_valid = _format_answer(safe_result)
    logger.info(
        "chat_completed session_id=%s has_sql=%s is_valid=%s",
        session_id,
        bool(final_sql),
        is_valid,
    )

    with _session_lock:
        history = _sessions.setdefault(session_id, [])
        history.append(
            {
                "role": "assistant",
                "content": answer,
                "created_at": _now(),
                "metadata": {
                    "sql": final_sql,
                    "generated_sql": generated_sql,
                    "explanation": explanation,
                    "is_valid": is_valid,
                },
            }
        )
        response_history = [ChatMessage(**item) for item in history]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sql=final_sql,
        generated_sql=generated_sql,
        explanation=explanation,
        is_valid=is_valid,
        history=response_history,
        raw_result=safe_result,
    )


@app.get("/sessions/{session_id}", response_model=list[ChatMessage])
def get_session(session_id: str) -> list[ChatMessage]:
    with _session_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return [ChatMessage(**item) for item in _sessions[session_id]]


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    with _session_lock:
        _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
