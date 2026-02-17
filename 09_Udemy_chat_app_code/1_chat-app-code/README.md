# OpenAI Agent AI Chat (Streamlit)

Gemini 기반 채팅봇을 Streamlit으로 실행하는 앱입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 다음 변수를 설정하세요.

- `GEMINI_API_KEY`: Google AI Studio에서 발급한 API 키 (필수)
- `GEMINI_MODEL`: 사용할 모델 이름 (선택, 기본값: `gemini-2.5-flash`)

예시는 `.env.example`을 참고하세요.

## 실행

```bash
streamlit run app.py
```

브라우저에서 표시되는 주소(기본 `http://localhost:8501`)로 접속하면 됩니다.

## 배포 (Streamlit Cloud 등)

API 키는 **Secrets**로 설정하세요. 예: `.streamlit/secrets.toml`에 다음 형식으로 추가할 수 있습니다.

```toml
GEMINI_API_KEY = "your-api-key"
GEMINI_MODEL = "gemini-2.5-flash"
```

앱에서 `os.getenv("GEMINI_API_KEY")`는 Streamlit Cloud의 secrets와 호환됩니다.
