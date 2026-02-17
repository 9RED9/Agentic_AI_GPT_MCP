# GPT와 MCP로 구현하는 자율형 AI Agent 구축 - 통합 교재 목차

이 문서는 Part I ~ Part III와 **01_~09_** 폴더 순서를 반영한 강의 진행 순서입니다.

---

## 전체 흐름

1. **Part I**: LangChain 없이 OpenAI Agent SDK만으로 Agent → RAG → Function Calling → MCP 구현
2. **Part II**: MCP 구현에 필요한 LangChain만 학습 (04_LangChain_MCP_Theory)
3. **Part III**: LangChain을 이용한 MCP 구현 (05_ → 06_ → 07_ → 08_ → 09_)

---

## Part I: OpenAI Agent SDK만으로 구현 (LangChain 없음)

이론 자료는 모두 `01_OpenAI_API_Basic/` 폴더 (01, 03, 05, 08, 11 등).

| 순서 | 주제 | 이론 (01_OpenAI_API_Basic) | 실습 | 비고 |
|------|------|----------------------|------|------|
| 1 | Agent | 01_Agents_Overview, 03_agent_runner | 01_OpenAI_API_Basic/10_chat-app-rag | Agent, Runner, 동기/비동기, 핸드오프·도구·가드레일·구조화출력 (01에 통합) |
| 2 | (보조) | 05_sessions, 08_error_handling | — | 세션·에러 핸들링 (필요 시) |
| 3 | RAG | 파일 검색/검색 도구 또는 Responses API | 01_OpenAI_API data 또는 09_/2_chat-app-rag-code | SDK 기반 검색·컨텍스트 |
| 4 | Function Calling | 01_Agents_Overview (도구 섹션) | 09_/3_chat-app-rag-function-calling 또는 01 내 도구 실습 | @function_tool, 도구 등록·호출 |
| 5 | MCP | 02_OpenAI_MCP_Agent_HTTP | 03_OpenAI_MCP_Agent_stdio (YouTube MCP + mcp.json + Streamlit) | 이론(HTTP)·실습(stdio), MCPServerStdio |
| 6 | (보조) | 11_Chatbot_RAG_Agent | 09_/code-mcp + 09_/4_chat-app-mcp-code, 10_chat-app-rag | 스트리밍·RAG·HTTP MCP 비교 |

---

## 02_OpenAI_MCP_Agent_HTTP (MCP 이론)

**위치**: `02_OpenAI_MCP_Agent_HTTP/`

Part I MCP **이론** 폴더. FastMCP HTTP 통합 서버(의류 + Chinook DB), Responses API로 MCP URL 연결.

- **mcp_local_server.py**: 통합 MCP 서버 (의류 + Chinook). 포트는 상단 `MCP_PORT` 상수. 기본 3000
- **responses_db_api.py**: Responses API + MCP server_url(HTTP), server_label `mcp_local_server`, mcp_approval 처리

**다음 실습**: stdio MCP + mcp.json + Streamlit은 **03_OpenAI_MCP_Agent_stdio**에서 진행.

---

## 03_OpenAI_MCP_Agent_stdio 실행 방법

**위치**: `03_OpenAI_MCP_Agent_stdio/` (기본 에이전트 실습은 `01_OpenAI_API_Basic/10_chat-app-rag` 참고)

### MCP + Streamlit (유튜브 에이전트)

- **클라이언트**: `mcp_client.py` — Streamlit UI + MCP 서버 연결 (mcp.json 기반)
- **MCP 서버**: `mcp_server.py` — FastMCP 기반 유튜브 도구(자막·검색·채널 정보), stdio
- **설정**: `mcp.json`에서 MCP 서버 경로 확인. `args`가 현재 폴더의 `mcp_server.py`를 가리키면 됨 (실행 시 작업 디렉터리는 `03_OpenAI_MCP_Agent_stdio`).

**실행 순서**

1. `03_OpenAI_MCP_Agent_stdio`를 작업 디렉터리로 한 터미널에서:
   ```bash
   streamlit run mcp_client.py
   ```
2. 브라우저에서 열리는 채팅 화면에서 유튜브 관련 질문 입력 (MCP 서버는 클라이언트가 stdio로 자동 실행).

**필요 환경**

- `.env`: `OPENAI_API_KEY`, 유튜브 검색용 `YOUTUBE_API_KEY` (mcp_server 도구 사용 시)
- 패키지: `agents`, `streamlit`, `python-dotenv`, `mcp`, `fastmcp`, `youtube-transcript-api`, `requests` 등
- Windows: asyncio 사용 시 `mcp_client.py` 내부에서 이미 처리됨

---

## Part II: MCP 구현에 필요한 LangChain 학습

**자료 위치**: `04_LangChain_MCP_Theory/` 폴더

| 순서 | 학습 내용 | 자료 | 비고 |
|------|-----------|------|------|
| 1 | LangChain 기본: 모델·메시지·컨텍스트 | 04_LangChain_MCP_Theory 내 "MCP 전제 지식" 노트 | init_chat_model, 메시지 구조 |
| 2 | LangChain 도구·에이전트: Tools & Agents | 동일 (101_Tools_Agents 스타일 참고) | create_agent, 도구 바인딩 |
| 3 | MCP 연동: MultiServerMCPClient, FastMCP | 04_LangChain_MCP_Theory/225_LangChain_MCP_Agent | MCP 개념, stdio/HTTP, LangChain 연동 |

---

## Part III: LangChain을 이용한 MCP 구현

**폴더 순서**: 05_ → 06_ → 07_ → 08_ → 09_ (번호 = 강의 순서)

| 순서 | 실습 폴더 | 비고 |
|------|------------|------|
| 1 | **(삭제) 05_Chinook_MCP_Lab** | Chinook MCP는 03_LangChain_Agent/400_DB_MCP_Agent 사용으로 통합됨 |
| 2 | **06_Notion_MCP_smithery** | 단일 Notion MCP (공식 @notionhq/notion-mcp-server + create_react_agent) |
| 3 | **07_Notion_DB_MCP_Total** | 다중 MCP: Chinook DB + Notion 통합. Chinook 서버는 03/400_DB_MCP_Agent 경로 사용 |
| 4 | **08_Streamlit_MCP_Lab** (선택) | Chinook MCP Streamlit UI → 03/400_DB_MCP_Agent의 streamlit_app.py 실행 |
| 5 | **09_Udemy_chat_app_code/5_agentic-app-langchain** | 종합: RAG(Chroma) + MCP + LangChain 에이전트 |

---

## 09_Udemy_chat_app_code 폴더 구조

**위치**: `09_Udemy_chat_app_code/` (Udemy 채팅 앱 코드, Part I·Part III 실습용)

| 폴더 | 역할 |
|------|------|
| **1_chat-app-code** | 기본 채팅 앱 (Gemini provider, app.py) |
| **2_chat-app-rag-code** | RAG 채팅 앱 (Gemini/OpenAI provider, rag.py, data/faqs·knowledgeBase) |
| **3_chat-app-rag-function-calling** | RAG + Function Calling (tools.py, data/customers·faqs·knowledgeBase) |
| **4_chat-app-mcp-code** | MCP 연동 채팅 앱 (MCP 서버 연동) |
| **5_agentic-app-langchain** | LangChain 에이전트 종합 (backend: agent, rag, tools, mcp_server / streamlit-app) |
| **code-mcp** | MCP 실습용 (agent/, tools/, mcp_server.py, data/customers·orders) |

루트: `README.md`, `project.toml`

---

## 폴더별 역할 요약 (01_~09_)

| 번호 | 폴더 | 역할 |
|------|------|------|
| 01 | 01_OpenAI_API_Basic | Part I 이론 + 기본 에이전트 실습 (10_chat-app-rag) |
| 02 | 02_OpenAI_MCP_Agent_HTTP | Part I 이론: MCP (HTTP 서버·Responses API·Agents SDK 예제) |
| 03 | 03_OpenAI_MCP_Agent_stdio | Part I 실습: MCP (YouTube MCP + mcp.json + Streamlit) |
| 04 | 04_LangChain_MCP_Theory | Part II 이론 (LangChain, MCP 연동) |
| 05 | (삭제) 05_Chinook_MCP_Lab | Chinook MCP는 03_LangChain_Agent/400_DB_MCP_Agent로 통합 |
| 06 | 06_Notion_MCP_smithery | 단일 Notion MCP + LangChain |
| 07 | 07_Notion_DB_MCP_Total | Chinook + Notion 다중 MCP 통합 |
| 08 | 08_Streamlit_MCP_Lab | Chinook MCP Streamlit 실습 (400_DB_MCP_Agent/streamlit_app.py) |
| 09 | 09_Udemy_chat_app_code | Udemy 채팅 앱 코드 (1_~5_ + code-mcp, Part I·Part III 실습) |
