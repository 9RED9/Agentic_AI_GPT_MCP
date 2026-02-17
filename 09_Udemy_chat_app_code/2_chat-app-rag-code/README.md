# OpenAI Agent AI Chat (Streamlit, RAG + OpenAI Agents SDK)

OpenAI Agents SDK와 RAG 도구(`@function_tool`)를 사용하는 채팅봇입니다. FAQ 데이터(`data/faqs.json`)를 임베딩하여 쿼리와 유사한 상위 문서를 검색하는 도구를 에이전트에 연결하고, 그 컨텍스트로 답변을 생성합니다. 사이드바에서 **Gemini**(LiteLLM) 또는 **OpenAI** 중 사용할 LLM을 선택할 수 있습니다.

## 설치

```bash
pip install streamlit python-dotenv pydantic
pip install "openai-agents[litellm]"
pip install openai google-genai
```

- `openai-agents[litellm]`: 에이전트 실행 + Gemini 사용 시 LiteLLM 연동
- Gemini 선택 시 RAG 검색용 임베딩에는 `google-genai`, OpenAI 선택 시에는 `openai` 임베딩 사용

## 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 다음 변수를 설정하세요.

- `GEMINI_API_KEY`: Google AI Studio에서 발급한 API 키 (Gemini 사용 시)
- `GEMINI_MODEL`: Gemini 모델 이름 (선택, 기본값: `gemini-2.5-flash`)
- `OPENAI_API_KEY`: OpenAI API 키 (OpenAI 사용 시)
- `OPENAI_MODEL`: OpenAI 모델 이름 (선택, 기본값: `gpt-5-nano`)

사이드바에서 선택한 LLM에 맞는 API 키만 설정하면 됩니다. 예시는 `.env.example`을 참고하세요.

## 데이터

`data/` 폴더에 다음 파일이 있어야 합니다.

- `faqs.json`: FAQ 목록 (RAG 검색용, 각 항목은 `question`, `answer` 포함)
- `knowledgeBase.json`: 지식 베이스 (선택적 사용)

## 실행

```bash
streamlit run app.py
```

브라우저에서 표시되는 주소(기본 `http://localhost:8501`)로 접속하면 됩니다.

## 배포 (Streamlit Cloud 등)

API 키는 **Secrets**로 설정하세요. 예: `.streamlit/secrets.toml`에 다음 형식으로 추가할 수 있습니다.

```toml
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-5-nano"
```

앱에서 `os.getenv()`로 읽으므로 Streamlit Cloud의 secrets와 호환됩니다.
