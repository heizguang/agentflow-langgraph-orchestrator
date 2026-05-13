import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.agent.langgraph_agent import ai_agent, checkpointer
from src.logger.logging import ERROR, INFO, log_event
from src.pinecone.vectorstore import add_doc_to_vectorstore
from utils.common import clean_text, generate_summary, generate_thread_id, load_pdf, retrieve_all_threads


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AgentFlow API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

thread_kb_meta: dict[str, dict] = {}
created_threads: list[str] = []


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


def make_seq_id() -> str:
    return f"seq-{uuid4().hex}"


def default_kb_meta() -> dict:
    return {
        "available": False,
        "topic": None,
        "summary": None,
    }


def ensure_thread(thread_id: str) -> None:
    if thread_id not in created_threads:
        created_threads.append(thread_id)
    thread_kb_meta.setdefault(thread_id, default_kb_meta())


def get_thread_kb_meta(thread_id: str) -> dict:
    ensure_thread(thread_id)
    return thread_kb_meta[thread_id]


def get_thread_messages(thread_id: str) -> list[dict]:
    ensure_thread(thread_id)

    try:
        state = ai_agent.get_state(config={"configurable": {"thread_id": thread_id}})  # type: ignore
        values = getattr(state, "values", {}) or {}
        messages = values.get("messages", [])
    except Exception:
        messages = []

    result: list[dict] = []
    for msg in messages:
        role = "assistant"
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        result.append({"role": role, "content": str(msg.content)})
    return result


def build_agent_input(message: str, thread_id: str) -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "external_kb_meta": get_thread_kb_meta(thread_id),
    }


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started_at
    log_event(INFO, f"{request.method} {request.url.path} status={response.status_code} elapsed={elapsed:.3f}s")
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/threads")
async def list_threads():
    saved_threads = [str(thread_id) for thread_id in retrieve_all_threads(checkpointer)]
    for thread_id in saved_threads:
        ensure_thread(thread_id)
    merged_threads = list(dict.fromkeys(created_threads + saved_threads))
    return {"threads": merged_threads[::-1]}


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    ensure_thread(thread_id)
    return {"thread_id": thread_id, "messages": get_thread_messages(thread_id)}


@app.post("/api/threads")
async def create_thread():
    thread_id = str(generate_thread_id())
    ensure_thread(thread_id)
    log_event(INFO, "新建会话", seq_id=make_seq_id(), thread_id=thread_id)
    return {"thread_id": thread_id}


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    message = (payload.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = (payload.thread_id or "").strip() or str(generate_thread_id())
    ensure_thread(thread_id)

    seq_id = make_seq_id()
    started_at = time.perf_counter()
    client_host = request.client.host if request.client else "-"

    log_event(
        INFO,
        "收到聊天请求",
        seq_id=seq_id,
        session_id=client_host,
        thread_id=thread_id,
    )

    try:
        response = ai_agent.invoke(
            build_agent_input(message, thread_id),
            config={"configurable": {"thread_id": thread_id}},
        )  # type: ignore

        agent_message = str(response["messages"][-1].content)
        elapsed = time.perf_counter() - started_at

        log_event(
            INFO,
            f"回答完成 elapsed={elapsed:.3f}s",
            seq_id=seq_id,
            session_id=client_host,
            thread_id=thread_id,
        )

        return {
            "thread_id": thread_id,
            "message": agent_message,
            "elapsed": round(elapsed, 3),
            "messages": get_thread_messages(thread_id),
        }

    except Exception as e:
        elapsed = time.perf_counter() - started_at
        log_event(
            ERROR,
            f"回答失败 elapsed={elapsed:.3f}s, error={e}",
            seq_id=seq_id,
            session_id=client_host,
            thread_id=thread_id,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    ensure_thread(thread_id)

    seq_id = make_seq_id()
    started_at = time.perf_counter()
    client_host = request.client.host if request.client else "-"

    log_event(
        INFO,
        f"开始处理 PDF file_name={file.filename}",
        seq_id=seq_id,
        session_id=client_host,
        thread_id=thread_id,
    )

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            temp_file_path = tmp_file.name

        documents = load_pdf(temp_file_path)
        if not documents:
            raise HTTPException(status_code=400, detail="无法解析 PDF 内容")

        extracted_text = "\n".join(doc.page_content for doc in documents)
        extracted_text = clean_text(extracted_text)

        summary = generate_summary(extracted_text)
        index_name = f"{thread_id}-{summary['topic']}"
        add_doc_to_vectorstore(index_name=index_name, content=extracted_text)

        thread_kb_meta[thread_id] = {
            "available": True,
            "topic": index_name,
            "summary": summary["summary"],
        }

        elapsed = time.perf_counter() - started_at
        log_event(
            INFO,
            f"PDF 处理完成 elapsed={elapsed:.3f}s",
            seq_id=seq_id,
            session_id=client_host,
            thread_id=thread_id,
        )

        return {
            "thread_id": thread_id,
            "topic": summary["topic"],
            "summary": summary["summary"],
            "elapsed": round(elapsed, 3),
        }

    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.perf_counter() - started_at
        log_event(
            ERROR,
            f"PDF 处理失败 elapsed={elapsed:.3f}s, error={e}",
            seq_id=seq_id,
            session_id=client_host,
            thread_id=thread_id,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
