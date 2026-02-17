# Notion MCP Smithery

## 프로그램 개요

### notion_agent_client.py
- **공식 Notion MCP 서버 연동**: @notionhq/notion-mcp-server (npx 실행)와 Chinook DB 서버를 통합
- **대화형 챗봇**: SQL 분석 후 Notion 공식 MCP 서버를 통해 페이지 생성 및 관리
- **환경변수 기반 설정**: NOTION_PAGE_ID를 기본 작업 페이지로 자동 설정
- **ReAct Agent**: LangGraph 기반으로 공식 Notion MCP 도구와 DB 분석 도구를 자동 선택

### Chinook.db
- **샘플 데이터베이스**: 음악 스트리밍 서비스의 고객, 아티스트, 앨범, 트랙 데이터
- **SQL 분석 대상**: 다양한 비즈니스 쿼리 연습 및 데이터 분석 예제용

## 주요 특징
- **공식 MCP 서버 사용**: Notion에서 공식 제공하는 @notionhq/notion-mcp-server 활용
- **npx 실행**: 별도 설치 없이 npm 패키지를 직접 실행하여 Notion API 연동
- **기본 페이지 설정**: 환경변수로 지정된 페이지를 기본 작업 공간으로 활용

## Chinook MCP 서버 경로 (03_LangChain_Agent/400_DB_MCP_Agent 사용)

이 폴더(07_Notion_DB_MCP_Total)에서는 Chinook DB MCP 서버를 subprocess로 띄웁니다. **03_LangChain_Agent/400_DB_MCP_Agent**의 `agent_server.py`를 사용합니다 (05_Chinook_MCP_Lab 삭제 후 통합).

- **경로**: `_project_root / "03_LangChain_Agent" / "400_DB_MCP_Agent" / "agent_server.py"`
- **Chinook.db**: `400_DB_MCP_Agent` 폴더 안에 두어야 합니다. 서버가 `__file__` 기준으로 같은 디렉터리에서 DB를 찾습니다.
