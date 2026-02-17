# Udemy Chat App Code (MCP + RAG + LLM)

이 저장소는 **단계별 학습**을 위해 하위 폴더마다 구현 방식을 다르게 두었습니다.  
처음에는 최소 구현으로 개념을 익히고, 마지막 폴더에서만 LangChain 기반의 실전 형태를 사용합니다.

---

## 왜 하위 폴더를 다르게 구성했는가 (교육 목적)

| 목적 | 설명 |
|------|------|
| **1~4, code-mcp** | **직접 구현**으로 흐름을 이해합니다. 프롬프트, RAG, 도구 호출, MCP를 “무슨 일이 일어나는지” 코드 수준에서 보여 줍니다. |
| **마지막 폴더(5_)** | **LangChain**으로 같은 기능을 구현해, 실무에서 쓰는 스택(벡터 DB·RAG·에이전트)을 한 번에 경험합니다. |

초반에 프레임워크를 쓰면 “무슨 일이 일어나는지”가 가려지므로,  
**먼저 원리(1~4, code-mcp) → 마지막에만 LangChain** 순서로 두었습니다.

---

## 폴더 구조와 역할

| 폴더 | 내용 | 스택 |
|------|------|------|
| **1_chat-app-code** | LLM과의 기본 채팅 (프롬프트/응답) | Python, Streamlit, Gemini |
| **2_chat-app-rag-code** | RAG 도입 (지식베이스·FAQ 검색 후 답변) | Python, Streamlit, JSON 기반 RAG |
| **3_chat-app-rag-function-calling** | RAG + 함수 호출 (도구 사용) | Python, Streamlit, 도구 직접 구현 |
| **4_chat-app-mcp-code** | MCP 클라이언트 (백엔드 API 호출) | Python, Streamlit |
| **code-mcp** | MCP 서버 (도구 노출, FastAPI + FastMCP) | Python, FastAPI, FastMCP |
| **5_agentic-app-langchain** | **종합 예제**: MCP + RAG(벡터 DB) + 에이전트 | **Python, LangChain, Chroma, FastMCP, Streamlit** |

---

## 실행 순서 (학습 추천)

1. **1_chat-app-code** → 채팅이 어떻게 동작하는지 확인  
2. **2_chat-app-rag-code** → RAG(컨텍스트 붙이기) 개념  
3. **3_chat-app-rag-function-calling** → 도구/함수 호출 흐름  
4. **code-mcp** → MCP 서버 실행 (`uvicorn app:app --port 8000`)  
5. **4_chat-app-mcp-code** → Streamlit에서 위 백엔드 호출  
6. **5_agentic-app-langchain** → LangChain으로 RAG·에이전트·MCP를 한 번에 사용  

---

## 환경 변수

각 하위 폴더의 `.env.example` 또는 README를 참고하세요.  
공통으로 쓰는 항목: `GEMINI_API_KEY`, `OPENAI_API_KEY`, (도구 사용 시) `WEATHER_API_KEY` 등.
