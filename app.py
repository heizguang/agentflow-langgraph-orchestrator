import os
import sys
import tempfile
import time
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage

from src.agent.langgraph_agent import ai_agent, checkpointer, external_kb_meta
from src.exception.exception_handler import AppException
from src.logger.logging import ERROR, INFO, log_event, logging
from src.pinecone.vectorstore import add_doc_to_vectorstore
from utils.common import clean_text, generate_summary, generate_thread_id, load_pdf, retrieve_all_threads

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:  # pragma: no cover
    get_script_run_ctx = None


st.set_page_config(
    page_title="AgentFlow",
    page_icon="🤖",
    layout="wide",
)


def get_session_id() -> str:
    if get_script_run_ctx is None:
        return "streamlit-session"

    ctx = get_script_run_ctx()
    if ctx is None:
        return "streamlit-session"

    session_id = getattr(ctx, "session_id", None)
    return f"session-{session_id}" if session_id else "streamlit-session"


def make_seq_id() -> str:
    return f"seq-{uuid4().hex}"


def reset_chat():
    try:
        new_thread_id = generate_thread_id()
        st.session_state["thread_id"] = new_thread_id
        add_thread(st.session_state["thread_id"])

        log_event(
            INFO,
            "新建会话",
            seq_id=make_seq_id(),
            session_id=get_session_id(),
            thread_id=str(new_thread_id),
        )

        st.session_state["chat_history"] = []
        st.session_state["pinecone_index"] = False
        st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1

    except Exception as e:
        logging.error(f"Error while resetting chat: {e}")
        raise AppException(e, sys)


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_chat_conversations(thread_id):
    try:
        return ai_agent.get_state(config={"configurable": {"thread_id": thread_id}}).values["messages"]  # type: ignore

    except Exception as e:
        logging.error(f"Error in loading chat conversations from thread_id: {e}")
        raise AppException(e, sys)


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "pinecone_index" not in st.session_state:
    st.session_state["pinecone_index"] = False

if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads(checkpointer)

if "external_kb_meta" not in st.session_state:
    st.session_state["external_kb_meta"] = {
        "available": False,
        "topic": None,
        "summary": None,
    }

add_thread(st.session_state["thread_id"])

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}


st.title("AgentFlow")
st.caption("多智能体助手，支持智能路由、文档问答和联网工具。")

if st.sidebar.button("新建对话"):
    reset_chat()

with st.sidebar:
    st.header("上传 PDF")
    uploaded_file = st.file_uploader("选择文件", type="pdf", key=st.session_state.get("upload_key"))

    if uploaded_file is not None and not st.session_state["pinecone_index"]:
        upload_seq_id = make_seq_id()
        upload_start = time.perf_counter()
        log_event(
            INFO,
            f"开始处理 PDF file_name={uploaded_file.name}",
            seq_id=upload_seq_id,
            session_id=get_session_id(),
            thread_id=str(st.session_state["thread_id"]),
        )

        with st.spinner("正在处理 PDF，请稍候..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name

            try:
                documents = load_pdf(temp_file_path)
                summary = None
                if documents:
                    extracted_text = "\n".join([doc.page_content for doc in documents])
                    extracted_text = clean_text(extracted_text)

                    summary = generate_summary(extracted_text)
                    add_doc_to_vectorstore(index_name=summary["topic"], content=extracted_text)

                    external_kb_meta["available"] = True
                    external_kb_meta["topic"] = summary["topic"]
                    external_kb_meta["summary"] = summary["summary"]

                    st.session_state["external_kb_meta"].update(
                        {
                            "available": True,
                            "topic": summary["topic"],
                            "summary": summary["summary"],
                        }
                    )

                st.session_state["pinecone_index"] = True
                elapsed = time.perf_counter() - upload_start
                log_event(
                    INFO,
                    f"PDF 处理完成 elapsed={elapsed:.3f}s",
                    seq_id=upload_seq_id,
                    session_id=get_session_id(),
                    thread_id=str(st.session_state["thread_id"]),
                )
                st.success("上传成功，已经写入本地知识库。")

            except Exception as e:
                elapsed = time.perf_counter() - upload_start
                log_event(
                    ERROR,
                    f"PDF 处理失败 elapsed={elapsed:.3f}s, error={e}",
                    seq_id=upload_seq_id,
                    session_id=get_session_id(),
                    thread_id=str(st.session_state["thread_id"]),
                )
                st.error("上传文件处理失败，请新建对话后重试。")
                raise AppException(e, sys)

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    st.header("历史会话")
    for thread_id in st.session_state["chat_threads"][::-1]:
        conv_id = f"对话-{str(thread_id)[:20]}"
        if st.button(conv_id):
            st.session_state["thread_id"] = thread_id
            log_event(
                INFO,
                "切换会话",
                seq_id=make_seq_id(),
                session_id=get_session_id(),
                thread_id=str(thread_id),
            )
            messages = load_chat_conversations(thread_id)

            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})

            st.session_state["chat_history"] = temp_messages


for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("请输入你的问题...")

if user_input:
    request_seq_id = make_seq_id()
    request_start = time.perf_counter()
    session_id = get_session_id()
    current_thread_id = str(st.session_state["thread_id"])

    log_event(
        INFO,
        "收到聊天请求",
        seq_id=request_seq_id,
        session_id=session_id,
        thread_id=current_thread_id,
    )

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "external_kb_meta": external_kb_meta,
    }

    try:
        response = ai_agent.invoke(initial_state, config=CONFIG)  # type: ignore
        agent_message = response["messages"][-1].content
        st.session_state["chat_history"].append({"role": "assistant", "content": agent_message})

        elapsed = time.perf_counter() - request_start
        log_event(
            INFO,
            f"回答完成 elapsed={elapsed:.3f}s",
            seq_id=request_seq_id,
            session_id=session_id,
            thread_id=current_thread_id,
        )

        with st.chat_message("assistant"):
            st.markdown(agent_message)

    except Exception as e:
        elapsed = time.perf_counter() - request_start
        log_event(
            ERROR,
            f"回答失败 elapsed={elapsed:.3f}s, error={e}",
            seq_id=request_seq_id,
            session_id=session_id,
            thread_id=current_thread_id,
        )
        raise
