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

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5.4-mini", model_provider="openai")

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
# 쿼리를 검토한 후 승인하면 중단됐던 DB 접근(`sql_db_query` 실행)이 재개되어 최종 답변까지 생성됩니다.

# %%
# 승인 예제 (실제로는 사용자가 검토 후 승인)
from langgraph.types import Command

result = agent_with_review.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
result['messages'][-1].pretty_print()

print("위의 Human-in-the-Loop 예제는 실제 환경에서 사용자가 쿼리를 검토하고 승인하는 과정을 보여줍니다.")

# %%

# %% [markdown]
# ### 실습 문제
#
# 본문에서 배운 **SQL 에이전트 + Human-in-the-Loop**를 활용해 다음 두 가지를 완성하세요.
#
# 1. **다양한 질의 패턴 실행**:
#    아래 4개의 질문을 리스트로 만들어 순회하면서 본문의 `agent`(`agent.invoke()`)로 실행하고,
#    각 질문과 최종 답변을 출력하세요.
#    - `"총 몇 개의 아티스트가 있나요?"` (단순 집계)
#    - `"가장 인기 있는 장르 상위 5개는?"` (조인 + 순위)
#    - `"고객별 총 구매 금액을 계산해주세요."` (그룹별 집계)
#    - `"앨범과 아티스트 정보를 함께 보여주세요."` (조인 조회)
#
# 2. **Human-in-the-Loop 승인 흐름 완성**:
#    본문 5장처럼 `HumanInTheLoopMiddleware`(`interrupt_on={"sql_db_query": True}`)와
#    `InMemorySaver` checkpointer를 가진 에이전트를 새로 만들고(`thread_id`는 새로 지정),
#    위 질문 중 하나를 실행해서 인터럽트로 멈추는 것을 확인한 뒤,
#    `Command(resume={"decisions": [{"type": "approve"}]})` 로 **승인하여 최종 답변까지** 출력하세요.
#    (본문 5.2에서는 주석으로만 보여준 승인 과정을 실제로 실행해 보는 문제입니다)
#
# ### 테스트 입력 예시
#
# * 1번: 4개 질문 각각에 대해 에이전트가 SQL을 생성/실행해 답변
#   👉 예: "총 몇 개의 아티스트가 있나요?" → 275개
#
# * 2번: `"고객별 총 구매 금액을 계산해주세요."`
#   👉 `sql_db_query` 실행 직전에 인터럽트 발생 → 승인(`approve`) 후 최종 답변 출력

# %%
