import os
import streamlit as st
from dotenv import load_dotenv

from gemini_provider import GeminiProvider
from openai_provider import OpenaiProvider
from rag import RagProvider

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None
if "faq_vectors_gemini" not in st.session_state:
    st.session_state.faq_vectors_gemini = None
if "faq_vectors_openai" not in st.session_state:
    st.session_state.faq_vectors_openai = None

st.set_page_config(page_title="OpenAI Agent AI Chat", layout="centered")

llm_choice = st.sidebar.selectbox(
    "LLM",
    ["Gemini", "OpenAI"],
    index=0,
    help="선택한 LLM에 따라 서로 다른 function calling API가 적용됩니다 (Gemini: functionDeclarations / OpenAI: tools + tool_calls).",
)

st.title("OpenAI Agent AI Chat")
st.caption("RAG + Function calling (날씨·고객 목록 도구)")

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
                    use_gemini = llm_choice == "Gemini"
                    rag = RagProvider()

                    if use_gemini:
                        if not GEMINI_API_KEY:
                            raise ValueError(
                                "GEMINI_API_KEY is not set. Add it to .env or environment."
                            )
                        provider = GeminiProvider(GEMINI_API_KEY, GEMINI_MODEL)
                        cache_key = "faq_vectors_gemini"
                        if st.session_state[cache_key] is None:
                            faq_data = rag.fetch_document_data("faqs.json")
                            answers = [item["answer"] for item in faq_data]
                            faq_embeddings = provider.generate_embeddings(
                                answers, task_type="RETRIEVAL_DOCUMENT"
                            )
                            st.session_state[cache_key] = [
                                {**faq_data[i], "vector": faq_embeddings[i]}
                                for i in range(len(faq_data))
                            ]
                        faq_vectors = st.session_state[cache_key]
                        query_embedding = provider.generate_embeddings(
                            prompt, task_type="RETRIEVAL_QUERY"
                        )
                    else:
                        if not OPENAI_API_KEY:
                            raise ValueError(
                                "OPENAI_API_KEY is not set. Add it to .env or environment."
                            )
                        provider = OpenaiProvider(OPENAI_API_KEY, OPENAI_MODEL)
                        cache_key = "faq_vectors_openai"
                        if st.session_state[cache_key] is None:
                            faq_data = rag.fetch_document_data("faqs.json")
                            answers = [item["answer"] for item in faq_data]
                            faq_embeddings = provider.generate_embeddings(answers)
                            st.session_state[cache_key] = [
                                {**faq_data[i], "vector": faq_embeddings[i]}
                                for i in range(len(faq_data))
                            ]
                        faq_vectors = st.session_state[cache_key]
                        query_embedding = provider.generate_embeddings(prompt)

                    query_vector = query_embedding[0]
                    rag_prompt = rag.prepare_rag_prompt(
                        prompt, query_vector, faq_vectors
                    )
                    reply = provider.generate_response_with_tools(rag_prompt)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.markdown(reply)
                except Exception as e:
                    st.session_state.error = str(e)
                    st.error(str(e))
