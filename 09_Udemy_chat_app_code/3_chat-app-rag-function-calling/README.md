# OpenAI Agent AI Chat (Streamlit, RAG + Function Calling)

RAG(FAQ 검색)와 **Function calling**을 함께 사용하는 채팅 앱입니다.  
사이드바에서 **Gemini** 또는 **OpenAI**를 선택하면, 같은 도구(날씨 조회·고객 목록)에 대해 **서로 다른 API 스펙**이 적용되는 것을 비교할 수 있습니다.

- **Gemini**: `functionDeclarations` → `functionCalls` → `functionResponse` (GenerateContentConfig tools)
- **OpenAI**: `tools` (type: function) → `tool_calls` → tool 메시지 (Chat Completions API)

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일에 다음을 설정하세요.

- `GEMINI_API_KEY`, `GEMINI_MODEL`: Gemini 사용 시
- `OPENAI_API_KEY`, `OPENAI_MODEL`: OpenAI 사용 시
- `WEATHER_API_KEY`: 날씨 도구(get_current_weather) 사용 시 (weatherapi.com)

예시는 `.env.example`을 참고하세요.

## 데이터

`data/` 폴더: `faqs.json`, `knowledgeBase.json`, `customers.json` (도구용).

## 실행

```bash
streamlit run app.py
```

브라우저에서 사이드바로 LLM을 바꾼 뒤, "서울 날씨 알려줘", "고객 목록 보여줘" 등으로 도구 호출을 테스트할 수 있습니다.
