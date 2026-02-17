import os
import streamlit as st
from dotenv import load_dotenv

from gemini_provider import GeminiProvider

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None

st.set_page_config(page_title="OpenAI Agent AI Chat", layout="centered")

st.title("OpenAI Agent AI Chat")
st.caption("Your AI assistant")

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
                    if not API_KEY:
                        raise ValueError("GEMINI_API_KEY is not set. Add it to .env or environment.")
                    provider = GeminiProvider(API_KEY, MODEL)
                    reply = provider.generate_response(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                except Exception as e:
                    st.session_state.error = str(e)
                    st.error(str(e))
