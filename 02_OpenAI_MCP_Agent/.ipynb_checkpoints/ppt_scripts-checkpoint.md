# PPT 스크립트 - 02_OpenAI_MCP_Agent (3장)

---

## 슬라이드 1: 실습 1 - mcp_1_stdio_client.py

**실습 1 - MCPServerStdio 클라이언트**

- Streamlit + agents의 MCPServerStdio로 mcp.json에 정의된 MCP 서버를 stdio로 연결하는 클라이언트 구현.

- MCP 서버: mcp_local_server.py → FastMCP + 의류 재고 + Chinook DB를 이용한 통합 서버 예제
  - 의류 도구 3개 (get_price, add_item, list_items) + Chinook DB 도구 3개 (execute_sql_query, list_tables, get_table_schema)

- 연결 방식: 클라이언트가 subprocess로 FastMCP 서버를 자동 기동하고, OpenAI 모델(Agent)이 그 서버의 의류·DB 도구를 사용.

- 적용 MCP 기능: name, async with 컨텍스트 매니저, cache_tools_list, create_static_tool_filter, mcp_config

- 챗봇: MCP 도구(의류 + Chinook DB)를 사용하는 챗봇을 Streamlit으로 작성.

---

## 슬라이드 2: 실습 2 - mcp_2_hosted_client.py

**실습 2 - HostedMCPTool 클라이언트**

- HostedMCPTool을 사용하여 OpenAI Responses API가 공개 MCP 서버를 직접 호출하는 방식.
  - 로컬 subprocess나 서버 관리가 전혀 불필요. OpenAI 인프라가 원격 MCP 서버에 직접 접속하여 도구 실행.

- GitMCP(gitmcp.io)를 활용하여 GitHub 저장소를 MCP 서버로 변환.
  - 저장소의 코드, README, 파일 구조를 도구로 제공.
  - URL 형식: https://gitmcp.io/{owner}/{repo}

- stdio와의 핵심 차이:
  - Agent(mcp_servers=[...]) 대신 Agent(tools=[HostedMCPTool(...)]) 로 전달
  - 서버 lifecycle 관리 불필요

- 챗봇: GitHub 저장소를 분석하는 챗봇을 Streamlit으로 작성.

---

## 슬라이드 3: 실습 3 - mcp_3_streamable_client.py

**실습 3 - MCPServerStreamableHttp + MCPServerManager 클라이언트**

- MCPServerStreamableHttp로 HTTP 기반 원격 MCP 서버에 연결.

- MCPServerManager로 다중 MCP 서버를 통합 관리.
  - 연결 성공/실패 서버 자동 분류 (active_servers / failed_servers)
  - 성공한 서버만 에이전트에 전달

- 고급 기능 적용:
  - tool_meta_resolver: 도구 호출 시 메타데이터(tenant_id 등) 자동 주입
  - dynamic tool_filter: 에이전트 이름에 따라 도구 동적 제한

- 실행 방법: python mcp_local_server.py --http 로 HTTP 서버 기동 후, Streamlit 앱에서 http://localhost:8000/mcp 로 연결.

- 챗봇: 다중 HTTP MCP 서버의 도구를 사용하는 챗봇을 Streamlit으로 작성.
