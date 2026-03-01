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
# # 05. Sessions
#
# **Session**은 여러 번의 `Runner.run` 호출에 걸쳐 **대화 히스토리를 자동으로 유지**하는 기능입니다.
# - 세션을 쓰지 않으면 매 턴마다 이전 메시지 목록을 직접 구성해 넘겨야 합니다.
# - `SQLiteSession(session_id)`를 사용하면 SDK가 대화를 DB에 저장/불러와서, 같은 `session`으로 실행할 때마다 맥락이 이어집니다.
#
# 활용: 채팅봇, 고객지원 에이전트 등 다회차 대화.
#
# | 세션 | 위치 | 특징 | 용도 |
# |---|---|---|---|
# | `SQLiteSession` | `agents` | 기본 내장, 인메모리/파일 저장 | 개발/프로토타입 |
# | `AsyncSQLiteSession` | `agents.extensions.memory` | 비동기 전용 SQLite | 비동기 환경 |
# | `AdvancedSQLiteSession` | `agents.extensions.memory` | 브랜치, 토큰 사용량 분석 | 고급 분석 필요 시 |
# | `SQLAlchemySession` | `agents.extensions.memory` | PostgreSQL, MySQL 등 지원 | 프로덕션 DB |
# | `RedisSession` | `agents.extensions.memory` | Redis 기반 분산 저장 | 대규모 서비스 |
# | `EncryptedSession` | `agents.extensions.memory` | 다른 세션을 암호화로 래핑 | 보안 필요 시 |
# | `OpenAIConversationsSession` | `agents` | OpenAI 서버에 대화 저장 | OpenAI 플랫폼 활용 |
# | `CustomSession` | 직접 구현 | `Session` 프로토콜 구현 | Django, MongoDB 등 |

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from agents import Agent, Runner, SQLiteSession

Model = "gpt-5-nano"

# %% [markdown]
# ## 세션 없이 실행 (맥락 없음)
#
# 세션 없이 두 번 호출하면, 두 번째 질문에 대한 맥락(이전 대화 내역과 연관된)이 없습니다.

# %%
agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model
)

# %%
question_1 = "남산 팔각정은 어느 도시에 있나요?"
result1 = await Runner.run(agent, question_1)
print("Q1:", question_1)
print("A1:", result1.final_output)

# %%
question_2 = "그 도시는 어느 나라에 있나요?"
result2 = await Runner.run(agent, question_2)
print("Q2:", question_2)
print("A2:", result2.final_output)

# %% [markdown]
# ## SQLiteSession으로 대화 이어가기
#
# 같은 `session` 인스턴스를 넘기면, 이전 턴의 메시지가 자동으로 포함됩니다.

# %%
session = SQLiteSession("conversation_123")

# %%
question_1 = "남산 팔각정은 어느 도시에 있나요?"
result1 = await Runner.run(
    agent, 
    question_1,
    session=session,
)
print("Q1:", question_1)
print("A1:", result1.final_output)

# %%
question_2 = "그 도시는 어느 나라에 있나요?"
result2 = await Runner.run(
    agent, 
    question_2,
    session=session
)
print("Q2:", question_2)
print("A2:", result2.final_output)

# %%
question_3 = "그곳 인구는 얼마인가요?"
result3 = await Runner.run(
    agent, 
    question_3,
    session=session
)
print("Q3:", question_3)
print("A3:", result3.final_output)

# %% [markdown]
# ## 세션 ID별로 대화 분리
#
# `session_id`가 다르면 서로 다른 대화로 저장됩니다.
# 예: `SQLiteSession("user_456", "conversations.db")`로 사용자/채널별 분리.

# %%
session_b = SQLiteSession("another_user_345")
result_b = await Runner.run(agent, "그곳 인구는 얼마인가요?", session=session_b)
print("User B - A1:", result_b.final_output)

# %% [markdown]
# ## 정리
#
# - `SQLiteSession(session_id)` 또는 `SQLiteSession(session_id, "conversations.db")`: 파일 DB에 대화 저장
# - `Runner.run_sync(agent, input, session=session)`: 같은 session을 넘기면 히스토리 자동 유지
# - 세션 ID 전략: 사용자ID, 채널, 날짜 등을 조합해 대화 단위 구분
