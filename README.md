# GPT와 MCP로 구현하는 자율형 AI Agent 구축

OpenAI Agents SDK와 LangChain/LangGraph를 활용하여 자율형 AI Agent를 단계적으로 구현하는 학습 프로젝트입니다.

## 전체 폴더 구조

```
Agentic_AI_GPT_MCP/
├── 01_OpenAI_API_Basic/          # Part I: OpenAI Agents SDK 기초
│   ├── 01_Agents_Overview.py     # Agent, 핸드오프 기본, 도구, 가드레일, 구조화 출력
│   ├── 02_Handoffs.py            # 핸드오프 심화 (handoff() 함수, 콜백, 입력 필터 등)
│   ├── 03_agent_runner.py        # Agent 실행 (동기/비동기, Runner)
│   ├── 05_sessions.py            # 세션 관리
│   ├── 08_error_handling.py      # 에러 핸들링
│   ├── 09_Chatbot_RAG_Agent.py   # RAG 에이전트 구현 실습
│   └── 10_chat-app-rag/          # RAG 채팅 앱 (Streamlit)
├── 02_OpenAI_MCP_Agent/          # Part I: MCP 연동 (stdio + Streamlit)
│   ├── mcp_local_server.py       # FastMCP 서버 (의류 + Chinook DB)
│   ├── mcp_client.py             # Streamlit UI 클라이언트
│   └── mcp.json                  # MCP 서버 자동 기동 설정
├── 03_LangChain_Agent/           # Part II/III: LangChain + LangGraph
│   ├── 101_Tools_Agents.py
│   ├── 102_Memory_Concepts.py
│   ├── 110_Semantic_Search.py
│   ├── 120_RAG_Agent.py
│   ├── 210_Data_Analysis_Agent.py
│   ├── 220_SQL_Agent.py
│   ├── 310_Subagents_Pattern.py
│   ├── 320_Handoffs_Pattern.py
│   ├── 330_Router_Pattern.py
│   ├── 340_Skills_Pattern.py
│   ├── 350_Human_In_The_Loop.py
│   ├── streamlit-llm_LangChain/  # Streamlit 실습 앱 모음
│   ├── 400_DB_MCP_Agent/         # Chinook DB MCP 서버 & 클라이언트
│   ├── 410_Notion_MCP_smithery/  # Notion 전용 MCP Agent
│   └── 420_Notion_DB_MCP_Total/  # Notion + Chinook 다중 MCP 통합
└── 04_Deep_Agent/                # Deep Agent (계획·코드 실행·아티팩트 생성)
    └── 010_Deep_Agent.ipynb
```

---

## Part I: OpenAI Agent SDK만으로 구현 (LangChain 없음)

### 1. 01_OpenAI_API_Basic — Agent 기초 + RAG + Function Calling

**이론 파일**

| 파일 | 내용 |
|------|------|
| `01_Agents_Overview.py` | Agent, 핸드오프 기본 개념, 도구(Function/Agents as tools), 가드레일, 구조화 출력 |
| `02_Handoffs.py` | 핸드오프 심화 — `handoff()` 함수, `on_handoff` 콜백, `input_type`, `input_filter`, 권장 프롬프트 |
| `03_agent_runner.py` | Agent 실행 (동기/비동기, Runner) |
| `05_sessions.py` | 세션 관리 |
| `08_error_handling.py` | 에러 핸들링 |
| `09_Chatbot_RAG_Agent.py` | RAG 에이전트 구현 실습 |

**실습 앱: `10_chat-app-rag/`**

OpenAI Agents SDK와 RAG 도구(`@function_tool`)를 사용하는 Streamlit 채팅봇입니다.
FAQ 데이터(`data/faqs.json`)를 임베딩하여 유사 문서를 검색하고, 그 컨텍스트로 답변을 생성합니다.
사이드바에서 **Gemini**(LiteLLM) 또는 **OpenAI** 중 LLM을 선택할 수 있습니다.

```bash
cd 01_OpenAI_API_Basic/10_chat-app-rag
streamlit run app.py
```

필요 패키지:
```bash
pip install streamlit python-dotenv pydantic
pip install "openai-agents[litellm]"
pip install openai google-genai
```

환경 변수 (`.env`):
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini            # 선택, 기본값: gpt-4o-mini
GEMINI_API_KEY=your_gemini_api_key  # Gemini 선택 시
GEMINI_MODEL=gemini-2.5-flash       # 선택
```

---

### 2. 02_OpenAI_MCP_Agent — MCP 연동 (stdio)

MCP 서버(의류 + Chinook DB)를 **stdio** 방식으로 사용하는 실습입니다.

**구성 파일**

| 파일 | 역할 |
|------|------|
| `mcp_local_server.py` | FastMCP 서버 — 의류(get_price, add_item, list_items) + Chinook(execute_sql_query, list_tables, get_table_schema) |
| `mcp.json` | 클라이언트가 서버를 subprocess로 자동 기동하는 설정 |
| `mcp_client.py` | Streamlit UI 클라이언트 (권장) |

**실행 방법**

```bash
# 작업 디렉터리: 02_OpenAI_MCP_Agent
streamlit run mcp_client.py
```

MCP 서버는 `mcp.json` 설정에 따라 클라이언트가 자동으로 실행합니다. 별도 터미널에서 서버를 먼저 실행할 필요 없습니다.

**필요 환경**
- 동일 폴더에 `Chinook.db`
- `.env`: `OPENAI_API_KEY`
- 패키지: `openai`, `agents`, `streamlit`, `fastmcp`, `python-dotenv`

---

## Part II/III: LangChain + LangGraph 에이전트

### 3. 03_LangChain_Agent — LangChain 3일 과정

LangChain과 LangGraph를 활용한 AI 에이전트 개발 과정입니다.

#### 커리큘럼

**1일차: 기초 개념 및 RAG**

| 파일 | 내용 |
|------|------|
| `101_Tools_Agents.py` | Tools & Agents 기초 |
| `102_Memory_Concepts.py` | 단기 메모리 개념 및 관리 |
| `110_Semantic_Search.py` | 시맨틱 검색 엔진 구축 |
| `120_RAG_Agent.py` | RAG 에이전트 구축 |

**2일차: 데이터 분석 및 SQL**

| 파일 | 내용 |
|------|------|
| `210_Data_Analysis_Agent.py` | 데이터 분석 에이전트 |
| `220_SQL_Agent.py` | SQL 데이터베이스 Q&A 에이전트 |

**3일차: 멀티 에이전트 패턴**

| 파일 | 내용 |
|------|------|
| `310_Subagents_Pattern.py` | 하위 에이전트 위임 패턴 |
| `320_Handoffs_Pattern.py` | 에이전트 간 핸드오프 패턴 |
| `330_Router_Pattern.py` | 라우터 패턴 |
| `340_Skills_Pattern.py` | 스킬 기반 패턴 |
| `350_Human_In_The_Loop.py` | Human-in-the-Loop 패턴 |

#### Streamlit 실습 앱 (`streamlit-llm_LangChain/`)

| 파일 | 연결 학습 | 실행 |
|------|-----------|------|
| `050_ChatGPT.py` | 기본 챗봇 (Models & Messages, Context Engineering) | `streamlit run 050_ChatGPT.py` |
| `060_Agent.py` | ReAct Agent (TavilySearch 웹 검색) | `streamlit run 060_Agent.py` |
| `110_Semantic_Search_App.py` | PDF 업로드 시맨틱 검색 엔진 | `streamlit run 110_Semantic_Search_App.py` |
| `120_RAG_Chatbot.py` | 웹 문서 기반 RAG 챗봇 | `streamlit run 120_RAG_Chatbot.py` |
| `210_Data_Analysis_App.py` | CSV 파일 자연어 데이터 분석 | `streamlit run 210_Data_Analysis_App.py` |
| `220_SQL_Chatbot.py` | SQL 챗봇 (Human-in-the-Loop 검토) | `streamlit run 220_SQL_Chatbot.py` |
| `310_Personal_Assistant_App.py` | 개인 비서 (Subagents, 캘린더·이메일) | `streamlit run 310_Personal_Assistant_App.py` |
| `350_Human_In_The_Loop_App.py` | 결제 승인 워크플로우 | `streamlit run 350_Human_In_The_Loop_App.py` |

실행 방법:
```bash
cd 03_LangChain_Agent/streamlit-llm_LangChain
streamlit run 050_ChatGPT.py
```

#### 환경 설정

```bash
# conda 환경 (권장)
conda env create -f environment.yml

# 또는 pip
pip install -r requirements.txt
```

```env
# .env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key  # 선택 (LangSmith 트레이싱)
LANGCHAIN_PROJECT=LangChain_V1            # 선택
```

---

### 4. 03_LangChain_Agent/400_DB_MCP_Agent — Chinook DB MCP Agent

Model Context Protocol(MCP)을 사용하여 Chinook 데이터베이스를 분석하는 대화형 AI 에이전트입니다.

**파일 구조**

```
400_DB_MCP_Agent/
├── agent_server.py      # Chinook DB MCP 서버 (FastMCP, stdio)
├── agent_client.py      # LangChain 기반 대화형 클라이언트 (콘솔)
├── streamlit_app.py     # Streamlit UI (agent_client의 Streamlit 버전)
└── Chinook.db           # 샘플 데이터베이스 (직접 준비)
```

**MCP 도구**

| 도구 | 설명 |
|------|------|
| `execute_sql_query(query)` | SQL 쿼리 실행 및 결과 반환 |
| `get_table_schema(table_name)` | 테이블 스키마(컬럼·타입·제약조건) 조회 |
| `list_tables()` | 전체 테이블 목록 반환 |

**실행 방법**

```bash
# 콘솔 클라이언트 (MCP 서버 자동 기동)
python agent_client.py

# Streamlit UI
streamlit run streamlit_app.py
```

**Chinook 데이터베이스** — 디지털 미디어 스토어 샘플 DB

| 주요 테이블 | 내용 |
|------------|------|
| Customer | 고객 정보 |
| Invoice / InvoiceLine | 주문 및 주문 상세 |
| Track / Album / Artist | 음악 트랙·앨범·아티스트 |
| Genre | 장르 정보 |
| Employee | 직원 정보 |

> **참고**: `420_Notion_DB_MCP_Total`에서 이 서버를 subprocess로 사용합니다.
> `Chinook.db` 파일은 `400_DB_MCP_Agent/` 폴더 안에 두어야 합니다 (서버가 `__file__` 기준으로 동일 디렉터리에서 DB를 탐색).

---

### 5. 03_LangChain_Agent/410_Notion_MCP_smithery — Notion 전용 MCP Agent

**파일**: `notion_agent_smithery_client.py`

공식 `@notionhq/notion-mcp-server`(npx 실행)만 사용하는 Notion 전용 대화형 챗봇입니다.

- LangGraph 기반 ReAct Agent
- 시작 시 사용 가능한 Notion MCP 도구 목록 자동 출력
- `NOTION_PAGE_ID` 환경변수로 기본 작업 페이지 지정

**환경 변수 (`.env`)**:

```env
OPENAI_API_KEY=your_openai_api_key
NOTION_API_KEY=your_notion_integration_token
NOTION_PAGE_ID=your_default_page_id
```

**실행**:

```bash
python notion_agent_smithery_client.py
```

---

### 6. 03_LangChain_Agent/420_Notion_DB_MCP_Total — Notion + Chinook 다중 MCP 통합

**파일**: `notion_agent_client.py`

공식 Notion MCP 서버와 Chinook DB MCP 서버를 동시에 연결하는 멀티 MCP 에이전트입니다.

- Chinook DB를 SQL로 분석하고, 결과를 Notion 페이지로 자동 생성
- Chinook MCP 서버 경로: `03_LangChain_Agent/400_DB_MCP_Agent/agent_server.py`

> **설정 필수**: `Chinook.db`를 `03_LangChain_Agent/400_DB_MCP_Agent/` 폴더에 배치하세요.

**환경 변수 (`.env`)**:

```env
OPENAI_API_KEY=your_openai_api_key
NOTION_API_KEY=your_notion_integration_token
NOTION_PAGE_ID=your_default_page_id
```

**실행**:

```bash
python notion_agent_client.py
```

---

## 심화: Deep Agent

### 7. 04_Deep_Agent — Deep Agent (계획·코드 실행·아티팩트 생성)

**파일**: `010_Deep_Agent.ipynb`

계획(Planning), 코드 실행(Code Execution), 아티팩트(스크립트·리포트·시각화 등) 생성이 필요한 복잡한 작업을 처리하기 위해 설계된 에이전트입니다.

- CSV 파일 분석 및 커스텀 도구 연동 예제 포함
- Backend: 샌드박스 환경에서 코드를 실행하는 백엔드 시스템
- Checkpointer: 멀티턴 대화를 지원하는 체크포인터

참고:
- [LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents)
- [LangChain Data Analysis](https://docs.langchain.com/oss/python/deepagents/data-analysis)

---

## 공통 환경 설정

프로젝트 루트 또는 각 폴더의 `.env` 파일에 API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key              # Gemini 사용 시
NOTION_API_KEY=your_notion_integration_token    # Notion MCP 사용 시
NOTION_PAGE_ID=your_default_notion_page_id      # Notion MCP 사용 시
LANGCHAIN_API_KEY=your_langchain_api_key        # LangSmith 트레이싱 시 (선택)
```

## 주의사항

- API 키는 반드시 `.env` 파일에 저장하고 버전 관리에 포함하지 마세요
- Chinook DB를 사용하는 실습(`02_OpenAI_MCP_Agent`, `400_DB_MCP_Agent`, `420_Notion_DB_MCP_Total`)은 `Chinook.db` 파일이 필요합니다
- SQL 에이전트 사용 시 보안을 고려하여 읽기 전용 권한을 권장합니다
- `.py`와 `.ipynb` 파일은 `jupytext`로 동기화됩니다 — `.py` 파일 수정 시 `.ipynb`가 자동 업데이트됩니다

## 참고 자료

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [LangChain 공식 문서](https://docs.langchain.com/oss/python/learn)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Notion MCP Server](https://github.com/notionhq/notion-mcp-server)
