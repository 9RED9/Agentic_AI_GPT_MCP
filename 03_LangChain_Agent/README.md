# LangChain 3일 과정 - 공식 튜토리얼 기반

LangChain과 LangGraph를 활용한 AI 에이전트 개발 과정

이 프로젝트는 [LangChain 공식 튜토리얼](https://docs.langchain.com/oss/python/learn)을 기반으로 구성된 3일 과정입니다.

## 📚 커리큘럼 개요

### 1일차: 기초 개념 및 RAG
**목표**: LangChain 기초 개념과 RAG 패턴 학습

| 파일 | 내용 | 공식 문서 |
|------|------|----------|
| `100_Models_Messages_Context.py` | Models, Messages, Context Engineering 기초 | [LangChain Fundamentals](https://docs.langchain.com/oss/python/learn) |
| `101_Tools_Agents.py` | Tools & Agents 기초 | [LangChain Fundamentals](https://docs.langchain.com/oss/python/learn) |
| `102_Memory_Concepts.py` | 단기 메모리 개념 및 관리 | [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) |
| `110_Semantic_Search.py` | 시맨틱 검색 엔진 구축 | [Semantic Search](https://docs.langchain.com/oss/python/langchain/knowledge-base) |
| `120_RAG_Agent.py` | RAG 에이전트 구축 | [RAG Agent](https://docs.langchain.com/oss/python/langchain/rag) |

### 2일차: 데이터 분석 및 LangGraph
**목표**: 데이터 분석, SQL 에이전트, MCP, LangGraph 활용

| 파일 | 내용 | 공식 문서 |
|------|------|----------|
| `210_Data_Analysis_Agent.py` | 데이터 분석 에이전트 | [Deep Agents Data Analysis](https://docs.langchain.com/oss/python/deepagents/data-analysis) |
| `220_SQL_Agent.py` | SQL 데이터베이스 Q&A 에이전트 | [SQL Agent](https://docs.langchain.com/oss/python/langchain/sql-agent) |
| `225_MCP_Agent.py` | Model Context Protocol (MCP) 기초 | [MCP](https://docs.langchain.com/oss/python/langchain/mcp) |
| `230_Custom_RAG_LangGraph.py` | LangGraph로 커스텀 RAG 구현 | [Custom RAG with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag) |
| `240_Custom_SQL_LangGraph.py` | LangGraph로 커스텀 SQL 에이전트 | [Custom SQL with LangGraph](https://docs.langchain.com/oss/python/langgraph/sql-agent) |

### 3일차: 멀티 에이전트 패턴
**목표**: 다양한 멀티 에이전트 패턴 및 Human-in-the-Loop

| 파일 | 내용 | 공식 문서 |
|------|------|----------|
| `310_Subagents_Pattern.py` | 하위 에이전트 위임 패턴 | [Subagents Personal Assistant](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant) |
| `320_Handoffs_Pattern.py` | 에이전트 간 핸드오프 패턴 | [Handoffs Customer Support](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support) |
| `330_Router_Pattern.py` | 라우터 패턴 | [Router Knowledge Base](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base) |
| `340_Skills_Pattern.py` | 스킬 기반 패턴 | [Custom Middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) |
| `350_Human_In_The_Loop.py` | Human-in-the-Loop 패턴 | [Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) |

## 🚀 시작하기

### 환경 설정

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **환경 변수 설정**
`.env` 파일을 생성하고 다음 변수를 설정하세요:
```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key  # 선택적
LANGCHAIN_PROJECT=LangChain_V1  # 선택적
```

3. **Jupyter Notebook 실행**
```bash
jupyter notebook
```

또는 Python 스크립트로 직접 실행:
```bash
python 100_Models_Messages_Context.py
```

## 📖 학습 방법

### 파일 형식
- 각 파일은 Jupyter Notebook (`.ipynb`)과 Python 스크립트 (`.py`)로 제공됩니다
- `jupytext`를 사용하여 자동 동기화됩니다
- `.py` 파일을 수정하면 `.ipynb` 파일이 자동으로 업데이트됩니다

### 학습 순서 및 Streamlit 실습 타이밍

#### 1일차: 기초 개념 및 RAG
1. **`100_Models_Messages_Context.py`** 학습
   - Models와 Messages 기본 개념
   - Context Engineering 개념 이해
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/050_ChatGPT.py` 실행
     ```bash
     cd streamlit-llm_LangChain
     streamlit run 050_ChatGPT.py
     ```

2. **`101_Tools_Agents.py`** 학습
   - Tools와 Agents 기본 개념
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/060_Agent.py` 실행
     ```bash
     streamlit run 060_Agent.py
     ```
   
3. **`102_Memory_Concepts.py`** 학습
   - 단기 메모리 개념 및 관리 방법 학습

4. **`110_Semantic_Search.py`** 학습
   - 시맨틱 검색 엔진 구축 방법 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/110_Semantic_Search_App.py` 실행
     ```bash
     streamlit run 110_Semantic_Search_App.py
     ```

5. **`120_RAG_Agent.py`** 학습
   - RAG 에이전트 구축 방법 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/120_RAG_Chatbot.py` 실행
     ```bash
     streamlit run 120_RAG_Chatbot.py
     ```

#### 2일차: 데이터 분석 및 LangGraph
1. **`210_Data_Analysis_Agent.py`** 학습
   - 데이터 분석 에이전트 구축 방법 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/210_Data_Analysis_App.py` 실행
     ```bash
     streamlit run 210_Data_Analysis_App.py
     ```

2. **`220_SQL_Agent.py`** 학습
   - SQL 에이전트 구축 방법 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/220_SQL_Chatbot.py` 실행
     ```bash
     streamlit run 220_SQL_Chatbot.py
     ```

3. **`225_MCP_Agent.py`** 학습
   - Model Context Protocol (MCP) 개념 및 활용
   - FastMCP로 MCP 서버 생성
   - MultiServerMCPClient로 MCP 도구 사용
   - MCP 기반 SQL 에이전트 구현

4. **`230_Custom_RAG_LangGraph.py`** 학습
   - LangGraph로 커스텀 RAG 구현 방법 학습

5. **`240_Custom_SQL_LangGraph.py`** 학습
   - LangGraph로 커스텀 SQL 에이전트 구현 방법 학습

#### 3일차: 멀티 에이전트 패턴
1. **`310_Subagents_Pattern.py`** 학습
   - 하위 에이전트 위임 패턴 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/310_Personal_Assistant_App.py` 실행
     ```bash
     streamlit run 310_Personal_Assistant_App.py
     ```

2. **`320_Handoffs_Pattern.py`** 학습
   - 에이전트 간 핸드오프 패턴 학습

3. **`330_Router_Pattern.py`** 학습
   - 라우터 패턴 학습

4. **`340_Skills_Pattern.py`** 학습
   - 스킬 기반 패턴 학습

5. **`350_Human_In_The_Loop.py`** 학습
   - Human-in-the-Loop 패턴 학습
   - ✅ **Streamlit 실습**: `streamlit-llm_LangChain/350_Human_In_The_Loop_App.py` 실행
     ```bash
     streamlit run 350_Human_In_The_Loop_App.py
     ```

### 실전 예제 활용 가이드
- **각 개념 학습 직후**: 해당하는 Streamlit 예제를 실행하여 실전 연습
- **예제 실행 전**: 관련 개념 파일을 먼저 학습하고 이해한 후 실행
- **예제 실행 후**: 코드를 수정하고 기능을 확장해보며 실전 경험 쌓기

## 🎯 Streamlit 실전 예제

### 1일차 예제
- **`streamlit-llm_LangChain/050_ChatGPT.py`** - 기본 챗봇 (Models & Messages + Context Engineering)
  - Streamlit Session 기반 대화 관리
  - 시스템 프롬프트 커스터마이징
  - Context Engineering 실습

- **`streamlit-llm_LangChain/060_Agent.py`** - ReAct Agent 챗봇 (Tools & Agents)
  - LangGraph ReAct Agent 구현
  - TavilySearch 도구 활용
  - 웹 검색 기반 질의응답

- **`streamlit-llm_LangChain/110_Semantic_Search_App.py`** - 시맨틱 검색 엔진
  - PDF 문서 업로드 및 인덱싱
  - 유사도 기반 검색
  - 검색 결과 시각화

- **`streamlit-llm_LangChain/120_RAG_Chatbot.py`** - RAG 챗봇
  - 웹 문서 인덱싱
  - 검색 기반 질의응답
  - 대화 이력 관리

### 2일차 예제
- **`streamlit-llm_LangChain/210_Data_Analysis_App.py`** - 데이터 분석 에이전트
  - CSV 파일 업로드
  - 자연어로 데이터 분석
  - 결과 시각화

- **`streamlit-llm_LangChain/220_SQL_Chatbot.py`** - SQL 챗봇
  - 자연어 SQL 쿼리 생성
  - Human-in-the-Loop 검토
  - 쿼리 결과 표시

### 3일차 예제
- **`streamlit-llm_LangChain/310_Personal_Assistant_App.py`** - 개인 비서 (Subagents)
  - 캘린더 일정 관리
  - 이메일 전송
  - 멀티 도메인 조율

- **`streamlit-llm_LangChain/350_Human_In_The_Loop_App.py`** - Human-in-the-Loop
  - 결제 승인 워크플로우
  - 승인/거부 인터페이스
  - 상태 관리

### Streamlit 실행 방법
```bash
cd streamlit-llm_LangChain
streamlit run 050_ChatGPT.py
```

## 📝 주요 개념

### 1일차 개념
- **컨텍스트 엔지니어링**: LLM에 적절한 컨텍스트 제공 방법
- **메모리**: 대화 이력 및 상태 관리
- **시맨틱 검색**: 의미 기반 문서 검색
- **RAG**: 검색 증강 생성 패턴

### 2일차 개념
- **데이터 분석**: Pandas DataFrame을 활용한 데이터 분석
- **SQL 에이전트**: 자연어로 SQL 쿼리 생성 및 실행
- **MCP (Model Context Protocol)**: 도구와 컨텍스트를 표준화된 방식으로 제공하는 프로토콜
- **LangGraph**: 복잡한 워크플로우를 그래프로 구성
- **커스텀 워크플로우**: Graph API를 활용한 복잡한 워크플로우 구성

### 3일차 개념
- **멀티 에이전트**: 여러 전문 에이전트를 조율하는 패턴
  - **Subagents**: 하위 에이전트 위임
  - **Handoffs**: 상태 기반 핸드오프
  - **Router**: 쿼리 라우팅
  - **Skills**: 동적 스킬 선택
- **Human-in-the-Loop**: 인간 개입이 필요한 작업 처리

## 🔗 참고 자료

- [LangChain 공식 문서](https://docs.langchain.com/oss/python/learn)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph)
- [LangChain Python API](https://api.python.langchain.com/)

## 📂 프로젝트 구조

```
LangChain_V1/
├── 1일차/
│   ├── 100_Models_Messages_Context.py
│   ├── 101_Tools_Agents.py
│   ├── 102_Memory_Concepts.py
│   ├── 110_Semantic_Search.py
│   └── 120_RAG_Agent.py
├── 2일차/
│   ├── 210_Data_Analysis_Agent.py
│   ├── 220_SQL_Agent.py
│   ├── 225_MCP_Agent.py
│   ├── 230_Custom_RAG_LangGraph.py
│   └── 240_Custom_SQL_LangGraph.py
├── 3일차/
│   ├── 310_Subagents_Pattern.py
│   ├── 320_Handoffs_Pattern.py
│   ├── 330_Router_Pattern.py
│   ├── 340_Skills_Pattern.py
│   └── 350_Human_In_The_Loop.py
├── streamlit-llm_LangChain/
│   ├── 050_ChatGPT.py
│   ├── 060_Agent.py
│   ├── 110_Semantic_Search_App.py
│   ├── 120_RAG_Chatbot.py
│   ├── 210_Data_Analysis_App.py
│   ├── 220_SQL_Chatbot.py
│   ├── 310_Personal_Assistant_App.py
│   └── 350_Human_In_The_Loop_App.py
└── README.md
```

## 🎓 학습 팁

1. **순차적 학습**: 파일 번호 순서대로 학습하는 것을 권장합니다
2. **실습 중심**: 코드를 직접 실행하고 수정해보세요
3. **Streamlit 활용**: 각 개념을 학습한 후 Streamlit 예제로 실전 연습
4. **공식 문서 참고**: 각 파일의 "참고" 링크에서 공식 문서 확인

## ⚠️ 주의사항

- API 키는 반드시 `.env` 파일에 저장하고 버전 관리에 포함하지 마세요
- 대용량 데이터셋을 사용할 때는 메모리 사용량을 주의하세요
- SQL 에이전트 사용 시 보안을 고려하여 읽기 전용 권한을 권장합니다

## 📄 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.
