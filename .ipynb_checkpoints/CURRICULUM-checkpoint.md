# GPT와 MCP로 구현하는 자율형 AI Agent 구축 - 통합 교재 목차

이 문서는 Part I ~ Part III와 **01_~08_** 폴더 순서를 반영한 강의 진행 순서입니다.

---

## 전체 흐름

1. **Part I**: LangChain 없이 OpenAI Agent SDK만으로 Agent → RAG → Function Calling → MCP 구현
2. **Part II**: MCP 구현에 필요한 LangChain만 학습 (03_LangChain_MCP_Theory)
3. **Part III**: LangChain을 이용한 MCP 구현 (04_ → 05_ → 06_ → 07_ → 08_)

---

## Part I: OpenAI Agent SDK만으로 구현 (LangChain 없음)

파일 번호 = 강의 순서 (01~12). 이론 자료는 모두 `01_OpenAI_API/` 폴더.

| 순서 | 주제 | 이론 (01_OpenAI_API) | 실습 | 비고 |
|------|------|----------------------|------|------|
| 1 | Agent | 01_Agents_Overview, 02_hello_world, 03_agent_runner | 02_Open_AI_Agent/openai_agent_basic | Agent, Runner, 동기/비동기 |
| 2 | (보조) | 04_handoffs, 05_sessions, 06~08 | — | 세션·가드레일·에러 (필요 시) |
| 3 | RAG | 파일 검색/검색 도구 또는 Responses API | 01_OpenAI_API data 또는 Udemy 2 (OpenAI) | SDK 기반 검색·컨텍스트 |
| 4 | Function Calling | 09_function_tools | Udemy 3 또는 09 실습 | @function_tool, 도구 등록·호출 |
| 5 | MCP | 10_mcp_tools, MCP/ | 02_Open_AI_Agent 전체 (YouTube MCP + mcp.json + Streamlit) | stdio MCP, MCPServerStdio |
| 6 | (보조) | 11_Chatbot_Agent, 12_tracing | Udemy code-mcp + 4_chat-app-mcp-code | 스트리밍·트레이싱·HTTP MCP 비교 |

---

## Part II: MCP 구현에 필요한 LangChain 학습

**자료 위치**: `03_LangChain_MCP_Theory/` 폴더

| 순서 | 학습 내용 | 자료 | 비고 |
|------|-----------|------|------|
| 1 | LangChain 기본: 모델·메시지·컨텍스트 | 03_LangChain_MCP_Theory 내 "MCP 전제 지식" 노트 | init_chat_model, 메시지 구조 |
| 2 | LangChain 도구·에이전트: Tools & Agents | 동일 (101_Tools_Agents 스타일 참고) | create_agent, 도구 바인딩 |
| 3 | MCP 연동: MultiServerMCPClient, FastMCP | 03_LangChain_MCP_Theory/225_LangChain_MCP_Agent | MCP 개념, stdio/HTTP, LangChain 연동 |

---

## Part III: LangChain을 이용한 MCP 구현

**폴더 순서**: 04_ → 05_ → 06_ → 07_ → 08_ (번호 = 강의 순서)

| 순서 | 실습 폴더 | 비고 |
|------|------------|------|
| 1 | **04_Chinook_MCP_Lab** | FastMCP 서버(Chinook) + LangChain 클라이언트(콘솔). 06_에서 Chinook 서버 참조 가능 |
| 2 | **05_Notion_MCP_smithery** | 단일 Notion MCP (공식 @notionhq/notion-mcp-server + create_react_agent) |
| 3 | **06_Notion_DB_MCP_Total** | 다중 MCP: Chinook DB + Notion 통합. Chinook 서버는 04_Chinook_MCP_Lab 경로 사용 |
| 4 | **07_Streamlit_MCP_Lab** (선택) | Chinook MCP + Streamlit UI |
| 5 | **08_Udemy_chat_app_code/5_agentic-app-langchain** | 종합: RAG(Chroma) + MCP + LangChain 에이전트 |

---

## 폴더별 역할 요약 (01_~08_)

| 번호 | 폴더 | 역할 |
|------|------|------|
| 01 | 01_OpenAI_API | Part I 이론 (Agent, RAG, Function Calling, MCP 등) |
| 02 | 02_Open_AI_Agent | Part I 실습: OpenAI Agent + MCP |
| 03 | 03_LangChain_MCP_Theory | Part II 이론 (LangChain, MCP 연동) |
| 04 | 04_Chinook_MCP_Lab | Chinook MCP 서버·클라이언트 (06_에서 참조) |
| 05 | 05_Notion_MCP_smithery | 단일 Notion MCP + LangChain |
| 06 | 06_Notion_DB_MCP_Total | Chinook + Notion 다중 MCP 통합 |
| 07 | 07_Streamlit_MCP_Lab | Streamlit 실습 (선택) |
| 08 | 08_Udemy_chat_app_code | Udemy 채팅 앱 코드 (5_agentic-app-langchain 등) |
