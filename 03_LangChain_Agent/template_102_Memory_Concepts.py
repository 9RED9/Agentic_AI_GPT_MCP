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
# # Memory Concepts (메모리 개념)
#
# **메모리(Memory)** 는 AI 에이전트가 대화 이력과 상태를 저장하고 관리하는 시스템입니다.
#
# 메모리는 두 가지 주요 유형으로 분류됩니다:
# - **단기 메모리(Short-term Memory)**: 하나의 대화 세션(thread) 내에서 이전 상호작용을 기억
# - **장기 메모리(Long-term Memory)**: 여러 대화 세션에 걸쳐 정보를 유지
#
# 이 노트북에서는 메모리의 기본 개념과 사용법을 다룹니다.
#
# **참고**: [LangChain 공식 문서 - Memory](https://docs.langchain.com/oss/python/concepts/memory)

# %%
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"로드된 API Key: {api_key[:12]}...{api_key[-4:]}")
else:
    print("❌ 여전히 API Key를 찾을 수 없습니다. 경로를 다시 확인해주세요.")

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5.4-mini", model_provider="openai")

# %% [markdown]
# ## 1. 단기 메모리 (Short-term Memory)
#
# 단기 메모리는 하나의 대화 세션(thread) 내에서 이전 메시지를 기억합니다.
# 에이전트가 대화의 문맥을 유지하고 이전 대화 내용을 참조할 수 있게 합니다.

# %% [markdown]
# ### 1.1 Checkpointer 없이 사용
#
# Checkpointer 없이는 각 호출이 독립적이며, 이전 대화를 기억하지 못합니다.

# %%
from langchain.agents import create_agent

# Checkpointer 없는 에이전트
agent_no_memory = create_agent(model, tools=[])

# 첫 번째 대화
result1 = agent_no_memory.invoke({
    "messages": [{"role": "user", "content": "내 이름은 홍길동이야."}]
})
print("응답 1:", result1["messages"][-1].content)

# 두 번째 대화 - 이전 내용을 기억하지 못함
result2 = agent_no_memory.invoke({
    "messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]
})
print("\n응답 2:", result2["messages"][-1].content)

# %% [markdown]
# ### 1.2 InMemorySaver로 단기 메모리 추가
#
# `InMemorySaver`는 메모리에 대화 이력을 저장합니다.
# 프로세스가 종료되면 데이터가 사라집니다.

# %%
# InMemorySaver를 사용한 에이전트
# 대화 설정 (thread_id로 대화 식별)
# 첫 번째 대화
# 두 번째 대화 - 같은 thread_id로 이전 내용을 기억

# %%
from langgraph.checkpoint.memory import InMemorySaver

# InMemorySaver를 사용한 에이전트
agent_with_memory = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 대화 설정 (thread_id로 대화 식별)
config = {"configurable": {"thread_id": "conversation_1"}}

# 첫 번째 대화
result1 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "내 이름은 홍길동이야."}]},
    config
)
print("응답 1:", result1["messages"][-1].content)

# 두 번째 대화 - 같은 thread_id로 이전 내용을 기억
result2 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
    config
)
print("\n응답 2:", result2["messages"][-1].content)

# 도구 호출 과정 포함 전체 메시지 확인
for msg in result2["messages"]:
    msg.pretty_print()

# %% [markdown]
# ## 2. 영구 저장 (Persistent Storage)
#
# `SqliteSaver`를 사용하면 대화 이력을 데이터베이스에 영구 저장할 수 있습니다.
# 프로세스가 재시작되어도 데이터가 유지됩니다.

# %% [markdown]
# ### 2.1 SqliteSaver 사용

# %%
# SQLite 데이터베이스 파일 경로
# SqliteSaver 생성 및 사용
    # 테이블 자동 생성
    # 에이전트 생성
    # 대화 진행
# 데이터베이스에서 다시 로드
    # 같은 thread_id로 이전 대화 이어가기

# %%
import os
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 데이터베이스 파일 경로
DB_PATH = "memory_checkpoints.db"

for f in ["memory_checkpoints.db", "memory_checkpoints.db-wal", "memory_checkpoints.db-shm"]:
    if os.path.exists(f):
        os.remove(f)

# SqliteSaver 생성 및 사용
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    # 테이블 자동 생성
    checkpointer.setup()
    
    # 에이전트 생성
    agent_persistent = create_agent(
        model,
        tools=[],
        checkpointer=checkpointer
    )
    
    # 대화 진행
    config = {"configurable": {"thread_id": "user_123"}}
    
    result = agent_persistent.invoke(
        {"messages": [{"role": "user", "content": "내 이메일은 hong@example.com이야."}]},
        config
    )
    print("저장 완료:", result["messages"][-1].content)

# %%
# 데이터베이스에서 다시 로드
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    agent_reload = create_agent(
        model,
        tools=[],
        checkpointer=checkpointer
    )
    
    # 같은 thread_id로 이전 대화 이어가기
    config = {"configurable": {"thread_id": "user_123"}}
    
    result = agent_reload.invoke(
        {"messages": [{"role": "user", "content": "내 이메일 주소가 뭐였지?"}]},
        config
    )
    print("\n재로드 후 응답:", result["messages"][-1].content)

# %% [markdown]
# ## 3. 대화 이력 조회
#
# Checkpointer를 통해 저장된 대화 이력을 조회할 수 있습니다.

# %%
# 여러 대화 진행
# 최종 응답 확인
# 전체 메시지 이력 확인

# %%
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[],
    checkpointer=checkpointer
)

# 여러 대화 진행
config = {"configurable": {"thread_id": "test_thread"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "안녕! 오늘 날씨가 좋네요."}]},
    config
)

agent.invoke(
    {"messages": [{"role": "user", "content": "점심으로 뭐 먹을까요?"}]},
    config
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "오늘 우리가 무슨 얘기했지?"}]},
    config
)

# 최종 응답 확인
result["messages"][-1].pretty_print()

# 전체 메시지 이력 확인
print("\n=== 전체 대화 이력 ===")
for i, msg in enumerate(result["messages"], 1):
    print(f"\n{i}. [{msg.type}] {msg.content[:100]}...")

# %%
# result
# result['messages']
# 도구 호출 과정 포함 전체 메시지 확인
for msg in result["messages"]:
    msg.pretty_print()

# %% [markdown]
# ---------------------------------
# ### 실전 예제: Flask 웹 애플리케이션
#
# 위에서 학습한 에이전트와 도구 사용법을 **Flask 웹 애플리케이션**으로 구현한 실전 예제를 참고하세요:
#
# - **`Flask_WebUI/101_Tools_Agent.py`** - ReAct Agent 기반 Chatbot 구현
#   - `create_agent`를 사용한 에이전트 생성
#   - 도구를 활용한 실시간 검색 및 응답
#   - 웹 UI를 통한 사용자 인터랙션
#   - memory 를 이용한 대화 내용 저장

# %% [markdown]
# # 실습 문제 모범답안 - 102. Memory Concepts
#
# `102_Memory_Concepts` 노트북 실습 문제에 대한 모범답안 예시입니다.
#
# ### 구현 요약
#
# 1. `InMemorySaver` 에이전트로 thread 격리 확인 — 같은 thread는 기억, 다른 thread는 모름
# 2. `SqliteSaver`로 대화를 `my_memory.db`에 영구 저장
# 3. checkpointer를 새로 열어 같은 thread_id로 대화 복원 확인

# %%
