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
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith 추적 (선택적 - API 키가 있을 때만 활성화)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano", model_provider="openai")

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
from langchain.tools import tool

@tool
def search_db(query: str) -> str:
    """데이터베이스에서 정보를 검색합니다."""
    return f"'{query}'에 대한 검색 결과를 찾았습니다."

# Checkpointer 없는 에이전트
agent_no_memory = create_agent(model, tools=[search_db])

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
from langgraph.checkpoint.memory import InMemorySaver

# InMemorySaver를 사용한 에이전트
agent_with_memory = create_agent(
    model,
    tools=[search_db],
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

# %% [markdown]
# ### 1.3 대화 세션 분리
#
# 다른 `thread_id`를 사용하면 별도의 대화 세션이 됩니다.

# %%
# 새로운 대화 세션
config_new = {"configurable": {"thread_id": "conversation_2"}}

result3 = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐야?"}]},
    config_new
)
print("새 세션 응답:", result3["messages"][-1].content)
print("(새 세션이므로 이전 이름을 기억하지 못함)")

# %% [markdown]
# ## 2. 영구 저장 (Persistent Storage)
#
# `SqliteSaver`를 사용하면 대화 이력을 데이터베이스에 영구 저장할 수 있습니다.
# 프로세스가 재시작되어도 데이터가 유지됩니다.

# %% [markdown]
# ### 2.1 SqliteSaver 사용

# %%
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 데이터베이스 파일 경로
DB_PATH = "memory_checkpoints.db"

# SqliteSaver 생성 및 사용
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    # 테이블 자동 생성
    checkpointer.setup()
    
    # 에이전트 생성
    agent_persistent = create_agent(
        model,
        tools=[search_db],
        checkpointer=checkpointer
    )
    
    # 대화 진행
    config = {"configurable": {"thread_id": "user_123"}}
    
    result = agent_persistent.invoke(
        {"messages": [{"role": "user", "content": "내 이메일은 hong@example.com이야."}]},
        config
    )
    print("저장 완료:", result["messages"][-1].content)

# 데이터베이스에서 다시 로드
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    agent_reload = create_agent(
        model,
        tools=[search_db],
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
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools=[search_db],
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

# %% [markdown]
# ## 4. 장기 메모리 개념 (Long-term Memory)
#
# 장기 메모리는 여러 대화 세션에 걸쳐 유지되는 정보입니다.
# 사용자 선호도, 이전 대화 요약, 학습한 정보 등을 저장합니다.
#
# 장기 메모리 구현 방법:
# - **Checkpointer 기반**: PostgreSQL, SQLite 등을 통한 대화 이력 저장
# - **Store 기반**: JSON 문서 형식으로 사용자 정보, 설정 등 저장
# - **벡터 데이터베이스**: 임베딩을 활용한 의미 기반 검색

# %% [markdown]
# ### 4.1 Checkpointer를 통한 여러 세션 정보 유지

# %%
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = "long_term_memory.db"

# 첫 번째 세션: 정보 저장
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    agent = create_agent(model, tools=[search_db], checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "user_alice"}}
    
    agent.invoke(
        {"messages": [{"role": "user", "content": "내 취미는 등산이야."}]},
        config
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "좋아하는 음식은 파스타야."}]},
        config
    )
    print("첫 번째 세션 완료")

# 두 번째 세션: 이전 정보 활용
with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
    agent = create_agent(model, tools=[search_db], checkpointer=checkpointer)
    
    # 같은 사용자의 새로운 대화
    config = {"configurable": {"thread_id": "user_alice"}}
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "내 취미와 좋아하는 음식을 다시 한번 알려줘."}]},
        config
    )
    print("\n두 번째 세션 응답:")
    result["messages"][-1].pretty_print()

# %% [markdown]
# ### 4.2 Store 기반 장기 메모리
#
# `Store`는 JSON 문서 형식으로 사용자 정보, 설정 등을 저장하는 장기 메모리입니다.
# Checkpointer와 달리 대화 이력이 아닌 구조화된 데이터를 저장합니다.
#
# 주요 개념:
# - **namespace**: 데이터를 그룹화하는 단위 (예: 사용자별, 조직별)
# - **key**: 각 메모리를 구분하는 고유 식별자
# - **value**: 저장할 JSON 문서

# %%
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import AgentState

# Store 생성
store = InMemoryStore()

# 사용자 정보 저장 예시
store.put(
    ("users",),  # namespace: 사용자 데이터 그룹
    "user_123",  # key: 사용자 ID
    {"name": "홍길동", "language": "한국어", "preferences": ["액션 영화", "이탈리안 음식"]}  # value: 사용자 정보
)

# 저장된 정보 조회
user_info = store.get(("users",), "user_123")
print("저장된 사용자 정보:", user_info.value)

# %% [markdown]
# ### 4.3 도구에서 Store 사용하기
#
# 도구에서 Store를 통해 장기 메모리에 접근할 수 있습니다.

# %%
@dataclass
class Context:
    user_id: str

@tool
def get_user_preferences(runtime: ToolRuntime[Context, AgentState]) -> str:
    """사용자의 선호도를 조회합니다."""
    store = runtime.store
    user_id = runtime.context.user_id
    user_info = store.get(("users",), user_id)
    if user_info:
        prefs = user_info.value.get("preferences", [])
        return f"사용자 선호도: {', '.join(prefs)}"
    return "사용자 정보를 찾을 수 없습니다."

# Store를 사용하는 에이전트 생성
agent_with_store = create_agent(
    model,
    tools=[get_user_preferences],
    store=store,
    context_schema=Context
)

# 사용자 선호도 조회
result = agent_with_store.invoke(
    {"messages": [{"role": "user", "content": "내 선호도를 알려줘"}]},
    context=Context(user_id="user_123")
)
result["messages"][-1].pretty_print()

# %% [markdown]
# ## 5. 메모리 전략
#
# 효과적인 메모리 관리를 위한 전략:
#
# ### 개발 단계
# - **InMemorySaver**: 빠른 프로토타이핑과 테스트
# - 데이터 손실이 문제되지 않는 경우
#
# ### 프로덕션 단계
# - **SqliteSaver**: 소규모 애플리케이션, 단일 인스턴스
# - **PostgresSaver**: 대규모 애플리케이션, 다중 인스턴스
# - 백업 및 복구 전략 필요
#
# ### 메모리 최적화
# - 오래된 대화 아카이빙
# - 중요 정보만 요약하여 저장
# - 컨텍스트 윈도우 제한 고려

# %% [markdown]
# ## 6. 실전 예제: 사용자 컨텍스트 유지

# %%
from langgraph.checkpoint.memory import InMemorySaver

@tool
def get_recommendation(category: str) -> str:
    """사용자의 선호도를 기반으로 추천을 제공합니다."""
    recommendations = {
        "영화": "액션 영화를 추천합니다: 미션 임파서블",
        "음식": "이탈리안 레스토랑을 추천합니다: 파스타 하우스",
        "책": "SF 소설을 추천합니다: 듄(Dune)"
    }
    return recommendations.get(category, "추천 정보가 없습니다.")

agent = create_agent(
    model,
    tools=[get_recommendation],
    checkpointer=InMemorySaver(),
    system_prompt="""당신은 개인화된 추천을 제공하는 AI 어시스턴트입니다.
사용자의 선호도를 기억하고, 이를 바탕으로 적절한 추천을 제공하세요."""
)

config = {"configurable": {"thread_id": "user_bob"}}

# 선호도 학습
agent.invoke(
    {"messages": [{"role": "user", "content": "나는 액션 영화를 좋아해."}]},
    config
)

agent.invoke(
    {"messages": [{"role": "user", "content": "이탈리안 음식도 좋아해."}]},
    config
)

# 컨텍스트 기반 추천
result = agent.invoke(
    {"messages": [{"role": "user", "content": "오늘 저녁에 뭐 하면 좋을까?"}]},
    config
)

result["messages"][-1].pretty_print()

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **InMemorySaver**: 개발/테스트용, 프로세스 종료 시 데이터 손실
# 2. **SqliteSaver**: 영구 저장, 소규모 애플리케이션에 적합
# 3. **Thread ID**: 대화 세션을 구분하는 고유 식별자
# 4. **Checkpointer**: 대화 상태를 저장하고 복원하는 메커니즘 (대화 이력 저장)
# 5. **Store**: JSON 문서 형식으로 사용자 정보, 설정 등을 저장 (구조화된 데이터 저장)
# 6. **장기 메모리**: 여러 세션에 걸친 정보 유지
#
# **다음 단계**: 
# - [110_Semantic_Search.py](110_Semantic_Search.py)에서 검색 엔진 구축 학습
# - 메모리와 검색을 결합한 RAG 에이전트 구현

# %%
