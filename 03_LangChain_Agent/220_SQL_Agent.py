# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SQL Agent (SQL 데이터베이스 Q&A 에이전트)
#
# 이 튜토리얼에서는 LangChain을 사용하여 SQL 데이터베이스에 대한 자연어 질의응답 시스템을 구축합니다.
# 에이전트는 자연어 질문을 SQL 쿼리로 변환하고, 결과를 바탕으로 답변을 생성합니다.
#
# **참고**: [LangChain 공식 문서 - SQL Agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
#
# ## 학습 내용
# - SQL 데이터베이스 연결
# - SQL 도구 생성
# - SQL 에이전트 구축
# - Human-in-the-loop 검토

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith 추적 (선택적)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano", model_provider="openai")

# %% [markdown]
# ## 1. 데이터베이스 연결
#
# SQLite 데이터베이스에 연결합니다. 이 예제에서는 Chinook 샘플 데이터베이스를 사용합니다.

# %% [markdown]
# ### 1.1 Chinook 데이터베이스 준비
#
# Chinook은 디지털 미디어 스토어를 나타내는 샘플 데이터베이스입니다.
#
# 데이터베이스가 없으면 다음 명령어로 생성할 수 있습니다:
# ```bash
# curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
# ```

# %%
from langchain_community.utilities import SQLDatabase

# SQLite 데이터베이스 연결
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 데이터베이스 정보 확인
print(f"데이터베이스 방언: {db.dialect}")
print(f"\n사용 가능한 테이블:")
print(db.get_usable_table_names())

# 샘플 쿼리 실행
sample_result = db.run("SELECT * FROM Artist LIMIT 5;")
print(f"\n샘플 쿼리 결과:\n{sample_result}")

# %% [markdown]
# ## 2. SQL 도구 생성
#
# SQLDatabaseToolkit을 사용하여 SQL 작업을 수행하는 도구를 생성합니다.

# %%
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# SQL 도구 키트 생성
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 사용 가능한 도구 확인
print("사용 가능한 SQL 도구:")
for tool in tools:
    print(f"\n- {tool.name}")
    print(f"  설명: {tool.description}")

# %% [markdown]
# ## 3. SQL 에이전트 생성
#
# SQL 도구를 사용하는 에이전트를 생성합니다.

# %%
from langchain.agents import create_agent

# 시스템 프롬프트 설정
system_prompt = """
당신은 SQL 데이터베이스와 상호작용하도록 설계된 에이전트입니다.
사용자로부터 질문을 입력받으면, 실행 가능한 {dialect} 구문으로
문법적으로 올바른 SQL 쿼리를 생성하세요.

그 후 쿼리를 실행한 결과를 확인하고, 그에 기반하여 답변을 반환하세요.
사용자가 명시적으로 원하는 예시 개수를 지정하지 않은 경우,
항상 최대 {top_k}개의 결과만 반환하도록 쿼리를 제한해야 합니다.

데이터베이스에서 가장 흥미로운 예시를 보여주기 위해,
관련 있는 열(column)을 기준으로 결과를 정렬할 수 있습니다.
특정 테이블의 모든 열을 조회하지 말고, 질문에 필요한 관련 열만 선택하세요.

쿼리를 실행하기 전에 반드시 구문을 두 번 확인하세요.
만약 쿼리 실행 중 오류가 발생하면, 쿼리를 다시 작성하고 재시도해야 합니다.

데이터베이스에 DML 문(INSERT, UPDATE, DELETE, DROP 등)을
절대 실행하지 마세요.

항상 데이터베이스에 어떤 테이블들이 존재하는지 먼저 확인한 뒤
어떤 데이터를 조회할 수 있는지 파악해야 합니다.
이 단계를 건너뛰지 마세요.

그 다음, 가장 관련성 높은 테이블의 스키마(schema)를 조회하세요.
""".format(
    dialect=db.dialect,
    top_k=5,
)

# SQL 에이전트 생성
agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)

# %% [markdown]
# ## 4. 에이전트 실행
#
# 자연어 질문으로 데이터베이스를 쿼리합니다.

# %%
# 질문 1: 집계 쿼리
question1 = "어떤 장르가 평균적으로 가장 긴 트랙을 가지고 있나요?"

print(f"질문: {question1}\n")
print("=" * 80)

for step in agent.stream(
    {"messages": [{"role": "user", "content": question1}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
# 질문 2: 테이블 설명
question2 = "playlisttrack 테이블의 구조를 설명해주세요."

print(f"\n질문: {question2}\n")
print("=" * 80)

for step in agent.stream(
    {"messages": [{"role": "user", "content": question2}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%
# 질문 3: 복잡한 조인 쿼리
question3 = "가장 많은 트랙을 가진 아티스트는 누구인가요?"

print(f"\n질문: {question3}\n")
print("=" * 80)

for step in agent.stream(
    {"messages": [{"role": "user", "content": question3}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %% [markdown]
# ## 5. Human-in-the-Loop 검토
#
# SQL 쿼리를 실행하기 전에 인간이 검토하고 승인할 수 있도록 설정합니다.
# 이는 보안과 정확성을 위해 중요합니다.

# %%
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# Human-in-the-loop 미들웨어가 있는 에이전트 생성
agent_with_review = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},  # sql_db_query 도구 호출 시 일시 정지
            description_prefix="도구 실행 승인 대기 중",
        ),
    ],
    checkpointer=InMemorySaver(),  # 상태 저장을 위한 체크포인터
)

# %% [markdown]
# ### 5.1 Human-in-the-Loop 실행
#
# 에이전트가 SQL 쿼리를 실행하기 전에 일시 정지하고 승인을 기다립니다.

# %%
question = "평균적으로 트랙 길이가 가장 긴 장르는 무엇인가요?"
config = {"configurable": {"thread_id": "review_1"}}

print(f"질문: {question}\n")
print("=" * 80)

# 스트리밍 실행
for step in agent_with_review.stream(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
    stream_mode="values",
):
    # 인터럽트 확인
    if "__interrupt__" in step:
        print("\n⚠️ HUMAN REVIEW REQUIRED ⚠️")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(f"\n대기 중인 작업:")
            print(f"도구: {request.get('tool', 'N/A')}")
            print(f"설명: {request.get('description', 'N/A')}")
            if 'args' in request:
                print(f"인자: {request['args']}")
        
        print("\n승인하려면 다음 코드를 실행하세요:")
        print("""
from langgraph.types import Command
result = agent_with_review.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
result['messages'][-1].pretty_print()
        """)
        break
    elif "messages" in step:
        step["messages"][-1].pretty_print()

# %% [markdown]
# ### 5.2 쿼리 승인 및 재개
#
# 쿼리를 검토한 후 승인하면 실행이 계속됩니다.

# %%
# 승인 예제 (실제로는 사용자가 검토 후 승인)
# from langgraph.types import Command
# 
# result = agent_with_review.invoke(
#     Command(resume={"decisions": [{"type": "approve"}]}),
#     config=config
# )
# result['messages'][-1].pretty_print()

print("위의 Human-in-the-Loop 예제는 실제 환경에서 사용자가 쿼리를 검토하고 승인하는 과정을 보여줍니다.")

# %% [markdown]
# ## 6. 보안 고려사항
#
# SQL 에이전트를 사용할 때는 다음 보안 사항을 고려해야 합니다:

# %% [markdown]
# ### 6.1 권한 제한
#
# - 읽기 전용 권한만 부여
# - 특정 테이블/스키마만 접근 가능하도록 제한
# - DML 문(INSERT, UPDATE, DELETE) 실행 방지

# %%
# 읽기 전용 연결 예제
# db_readonly = SQLDatabase.from_uri(
#     "sqlite:///Chinook.db",
#     include_tables=["Artist", "Album", "Track"],  # 특정 테이블만 접근
#     sample_rows_in_table_info=3  # 스키마 정보에 샘플 행 포함
# )

print("보안 모범 사례:")
print("1. 읽기 전용 데이터베이스 사용자 사용")
print("2. 필요한 테이블만 접근 가능하도록 제한")
print("3. Human-in-the-loop로 쿼리 검토")
print("4. 쿼리 실행 로그 모니터링")

# %% [markdown]
# ## 7. 실전 예제: 다양한 질의 패턴

# %%
# 질의 패턴 테스트
queries = [
    "총 몇 개의 아티스트가 있나요?",
    "가장 인기 있는 장르 상위 5개는?",
    "고객별 총 구매 금액을 계산해주세요.",
    "앨범과 아티스트 정보를 함께 보여주세요."
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"질문: {query}")
    print("="*80)
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print("\n답변:")
    result["messages"][-1].pretty_print()

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **SQLDatabase**: 데이터베이스 연결 및 쿼리 실행
# 2. **SQLDatabaseToolkit**: SQL 작업을 위한 도구 생성
# 3. **에이전트**: 자연어를 SQL로 변환하고 실행
# 4. **Human-in-the-Loop**: 쿼리 실행 전 검토 및 승인
# 5. **보안**: 읽기 전용 권한, 테이블 제한, 쿼리 검토
#
# **다음 단계**:
# - [225_MCP_Agent.py](225_MCP_Agent.py)에서 Model Context Protocol (MCP)를 사용한 SQL 에이전트 학습 
# - [240_Custom_SQL_LangGraph.py](240_Custom_SQL_LangGraph.py)에서 LangGraph로 더 세밀한 제어 구현
# - [350_Human_In_The_Loop.py](350_Human_In_The_Loop.py)에서 Human-in-the-Loop 패턴 자세히 학습
# - [streamlit-llm_LangChain/220_SQL_Chatbot.py](streamlit-llm_LangChain/220_SQL_Chatbot.py)에서 웹 UI 구현

# %%
