# OpenAI Agent AI Chat (Streamlit, MCP)

**MCP(Model Context Protocol)** 서버를 사용하는 채팅 앱입니다.  
RAG와 로컬 function calling 없이, MCP 서버의 도구(날씨·고객·주문)로만 응답합니다.

- 사이드바에서 **Gemini** 또는 **OpenAI**를 선택하면 MCP 서버가 해당 LLM과 도구로 응답합니다.
- MCP 서버는 `code-mcp/python-mcp`를 먼저 실행해야 합니다.

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일에 다음을 설정하세요.

- `MCP_SERVER_URL`: MCP 서버 주소 (기본값: `http://localhost:8000`)

예시는 `.env.example`을 참고하세요.  
(Gemini/OpenAI API 키는 MCP 서버 쪽 `.env`에 설정합니다.)

## 실행

1. MCP 서버를 먼저 실행합니다 (프로젝트 루트의 `code-mcp/python-mcp`).

   ```bash
   cd code-mcp/python-mcp && uvicorn app:app --reload
   ```

2. Streamlit 앱을 실행합니다.

   ```bash
   streamlit run app.py
   ```

브라우저에서 "서울 날씨 알려줘", "고객 목록 보여줘" 등으로 MCP 도구 호출을 테스트할 수 있습니다.
