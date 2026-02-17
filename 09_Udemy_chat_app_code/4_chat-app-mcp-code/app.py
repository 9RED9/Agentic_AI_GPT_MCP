import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000").rstrip("/")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None

st.set_page_config(page_title="OpenAI Agent AI Chat", layout="centered")

llm_choice = st.sidebar.selectbox(
    "LLM",
    ["Gemini", "OpenAI"],
    index=0,
    help="MCP 서버가 선택한 LLM과 도구(날씨·고객·주문)로 응답합니다.",
)

st.title("OpenAI Agent AI Chat")
st.caption("MCP (날씨·고객·주문 도구)")

for msg in st.session_state.messages:
    left_col, right_col = st.columns(2)
    if msg["role"] == "user":
        with right_col:
            with st.chat_message("user"):
                st.markdown(msg["content"])
    else:
        with left_col:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

if prompt := st.chat_input("Type your message..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    left_col, right_col = st.columns(2)
    with right_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    left_col2, right_col2 = st.columns(2)
    with left_col2:
        with st.chat_message("assistant"):
            with st.spinner("Bot is typing..."):
                try:
                    if not MCP_SERVER_URL:
                        raise ValueError(
                            "MCP_SERVER_URL is not set. Add it to .env (e.g. http://localhost:8000)."
                        )
                    model = "gemini" if llm_choice == "Gemini" else "openai"
                    resp = requests.post(
                        f"{MCP_SERVER_URL}/api/chat",
                        json={"message": prompt, "model": model},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    reply = data.get("reply", "").strip() or "(No reply)"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.markdown(reply)
                except requests.exceptions.RequestException as e:
                    err = str(e)
                    if hasattr(e, "response") and e.response is not None:
                        try:
                            err = e.response.json().get("detail", err)
                        except Exception:
                            err = e.response.text or err
                    st.session_state.error = err
                    st.error(err)
                except Exception as e:
                    st.session_state.error = str(e)
                    st.error(str(e))
