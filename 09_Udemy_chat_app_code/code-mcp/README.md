# Python MCP (Gemini + OpenAI)

code-mcp의 Node/Angular 기반 Gemini 전용 MCP를 Python으로 재구현한 버전입니다. 동일한 MCP 도구(weather, customer, order)를 제공하며, **Gemini**와 **OpenAI** 둘 다 선택할 수 있습니다.

## 구조

- **MCP 서버**: FastMCP로 6개 도구 노출 (getWeatherData, getCustomers, getCustomerById, getOrders, getOrdersWithCustomerDetails, getOrderById). FastAPI 앱에 `/mcp`로 마운트됩니다.
- **에이전트**: `POST /api/chat`에서 `model` 값에 따라 Gemini 또는 OpenAI로 요청합니다. 도구 호출 시 MCP와 동일한 스키마로 각 LLM API에 넘기고, 실행은 in-process로 동일 Python 함수를 사용합니다.

## 설치

```bash
cd code-mcp/python-mcp
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일에 다음을 설정하세요. 예시는 `.env.example` 참고.

- `GEMINI_API_KEY`, `GEMINI_MODEL`: Gemini 사용 시
- `OPENAI_API_KEY`, `OPENAI_MODEL`: OpenAI 사용 시
- `WEATHER_API_KEY`: getWeatherData 도구용 (weatherapi.com)
- `MCP_PORT`: MCP 서버만 별도 실행할 때 포트 (기본 8001)

## 실행

단일 프로세스로 FastAPI + MCP 서버 동시 실행:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- 채팅 API: `POST http://localhost:8000/api/chat`  
  Body: `{ "message": "서울 날씨 알려줘", "model": "gemini" }` 또는 `"model": "openai"`
- MCP 엔드포인트: `http://localhost:8000/mcp` (Streamable HTTP)

MCP 서버만 별도 실행하려면:

```bash
python mcp_server.py
```

(MCP_PORT 기본 8001에서 동작. 이때 에이전트는 동일 프로세스의 도구 함수를 사용하므로 FastAPI만 띄워도 채팅은 동작합니다.)

## 도구 목록 (Node와 동일)

| 도구 | 설명 |
|------|------|
| getWeatherData(city, country?) | weatherapi.com으로 날씨 조회 |
| getCustomers(limit?) | data/customers.json 목록 |
| getCustomerById(id) | ID로 고객 조회 |
| getOrders(limit?) | data/orders.json 목록 |
| getOrdersWithCustomerDetails(limit?) | 주문 + 고객명 |
| getOrderById(id) | ID로 주문 조회 |

## Angular 프론트 연동

기존 Angular 앱의 `environment.serverUrl`을 `http://localhost:8000`으로 두면 `POST /api/chat`으로 연동됩니다. 요청 시 `model`을 보내지 않으면 기본값 `gemini`가 사용됩니다.
