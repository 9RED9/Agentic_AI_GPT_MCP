# 02_OpenAI_MCP_Agent (stdio)

MCP 서버(의류 + Chinook DB)를 **stdio** 방식으로 사용.

## 구성

- **MCP 서버**: `mcp_local_server.py` — FastMCP, 의류(get_price, add_item, list_items) + Chinook(execute_sql_query, list_tables, get_table_schema)
- **설정**: `mcp.json` — 클라이언트가 서버를 subprocess로 자동 기동
- **클라이언트**:  
  - `mcp_client.py` — Streamlit UI (권장)  
  - `3_responses_clothing_db_api.py` — CLI 대화형

## 실행 순서

1. **Streamlit (권장)**  
   작업 디렉터리를 `02_OpenAI_MCP_Agent_HTTP`로 한 뒤:
   ```bash
   streamlit run mcp_client.py
   ```
   MCP 서버는 클라이언트가 `mcp.json`에 따라 자동으로 실행합니다. 별도 터미널에서 서버를 띄울 필요 없습니다.

2. **CLI**  
   ```bash
   python 3_responses_clothing_db_api.py
   ```

## 필요 환경

- 동일 폴더에 **Chinook.db** (Chinook 도구 사용 시)
- `.env`: `OPENAI_API_KEY`
- 패키지: `openai`, `agents`, `streamlit`, `fastmcp`, `python-dotenv` 등

## 노트북

- `1_responses_api_clothing.ipynb` — 의류 도구 예제 (stdio + Agents)
- `2_responses_api_chinook.ipynb` — Chinook DB 도구 예제 (stdio + Agents)

위 순서대로 셀만 실행하면 되며, MCP 서버는 노트북에서 사용하는 에이전트가 자동 기동합니다.
