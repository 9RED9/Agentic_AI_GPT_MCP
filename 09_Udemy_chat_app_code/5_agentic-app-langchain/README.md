# 5_agentic-app-langchain

이 폴더는 **LangChain**으로 MCP + RAG(벡터 DB) + 에이전트를 구현한 **종합 예제**입니다.  
프로젝트의 마지막 단계로, 실무에서 쓰는 스택(Chroma, LangChain 도구·에이전트)을 한 번에 사용합니다.

## 구조

- **backend**: FastAPI + FastMCP, LangChain 에이전트(도구 호출), LangChain Chroma RAG
- **streamlit-app**: 채팅 UI (백엔드 `POST /api/chat` 호출)

## 백엔드 (LangChain)

- **RAG**: `langchain-chroma` + `GoogleGenerativeAIEmbeddings`, `rag/rag_engine.py`(검색·프롬프트 조립), `rag/ingest.py`(문서 청킹·수집)
- **에이전트**: `ChatGoogleGenerativeAI` / `ChatOpenAI`에 `bind_tools`로 도구 7개(날씨·고객·주문·ragSearch) 연결, tool_calls 루프
- **MCP**: FastMCP로 동일 도구 노출 (`/mcp`)

## 설치 및 실행

### 1. 백엔드

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# .env에 GEMINI_API_KEY, OPENAI_API_KEY, (선택) WEATHER_API_KEY 설정
```

RAG 벡터 DB 초기화 (최초 1회):

```bash
cd backend
python -m rag.ingest data/rag_docs
```

서버 실행:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

- API: `POST http://localhost:8000/api/chat` (body: `{"message": "...", "model": "gemini"}` 또는 `"openai"`)
- MCP: `http://localhost:8000/mcp`

### 2. Streamlit

```bash
cd streamlit-app
pip install -r requirements.txt
cp .env.example .env
# .env에 BACKEND_URL=http://localhost:8000 (기본값)
streamlit run app.py
```

## 환경 변수 (backend)

| 변수 | 설명 |
|------|------|
| GEMINI_API_KEY | Gemini API 키 (RAG 임베딩·채팅) |
| GEMINI_MODEL | 채팅 모델 (기본 gemini-2.5-flash) |
| OPENAI_API_KEY | OpenAI API 키 |
| OPENAI_MODEL | 채팅 모델 (기본 gpt-5-nano) |
| WEATHER_API_KEY | weatherapi.com 키 (getWeatherData 도구) |
| CHROMA_PERSIST_DIR | Chroma 저장 경로 (기본 vector_data) |
| CHROMA_COLLECTION | 컬렉션 이름 (기본 rag_documents) |
