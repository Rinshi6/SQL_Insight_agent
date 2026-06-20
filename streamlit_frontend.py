from __future__ import annotations

import json
import os
from datetime import datetime
from uuid import uuid4

import requests
import streamlit as st


API_URL = os.getenv("TXT2SQL_API_URL", "http://localhost:3000").rstrip("/")


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat() -> None:
    session_id = st.session_state.session_id
    try:
        requests.delete(f"{API_URL}/sessions/{session_id}", timeout=10)
    except requests.RequestException:
        pass

    st.session_state.session_id = uuid4().hex
    st.session_state.messages = []


def chat_as_markdown() -> str:
    lines = [
        "# Text-to-SQL Chat",
        f"- Session: `{st.session_state.session_id}`",
        f"- Exported: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
    ]

    for item in st.session_state.messages:
        role = item["role"].title()
        lines.append(f"## {role}")
        lines.append(item["content"])
        lines.append("")

    return "\n".join(lines)


def chat_as_json() -> str:
    payload = {
        "session_id": st.session_state.session_id,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    return json.dumps(payload, indent=2)


def ask_backend(message: str) -> dict:
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": message, "session_id": st.session_state.session_id},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Text-to-SQL Agent", page_icon="SQL", layout="wide")
init_state()

with st.sidebar:
    st.header("Session")
    st.caption(st.session_state.session_id)

    if st.button("Clear chat", use_container_width=True):
        clear_chat()
        st.rerun()

    st.download_button(
        "Download chat",
        data=chat_as_markdown(),
        file_name=f"txt2sql-chat-{st.session_state.session_id}.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    st.download_button(
        "Download JSON",
        data=chat_as_json(),
        file_name=f"txt2sql-chat-{st.session_state.session_id}.json",
        mime="application/json",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    st.divider()
    st.caption(f"Backend: {API_URL}")

st.title("Text-to-SQL Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the database")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            try:
                result = ask_backend(prompt)
                st.session_state.session_id = result["session_id"]
                answer = result["answer"]
            except requests.RequestException as exc:
                answer = f"Backend request failed: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
