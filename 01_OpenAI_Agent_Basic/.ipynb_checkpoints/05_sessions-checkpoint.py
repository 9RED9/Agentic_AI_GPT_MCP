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
# - `session`을 넘기면 SDK가 대화를 저장소에 저장/불러와서, 같은 `session`으로 실행할 때마다 맥락이 이어집니다.
#
# 활용: 채팅봇, 고객지원 에이전트 등 다회차 대화.
#
# ### SDK 제공 Session 유형 (10종)
#
# | 세션 | 위치 | 특징 | 용도 | 본 노트북 실습 |
# |---|---|---|---|---|
# | `SQLiteSession` | `agents` | 기본 내장, 인메모리/파일 저장 | 개발/프로토타입 | ✅ |
# | `AsyncSQLiteSession` | `agents.extensions.memory` | 비동기 전용 SQLite | 비동기 환경 | ✅ |
# | `AdvancedSQLiteSession` | `agents.extensions.memory` | 브랜치, 토큰 사용량 분석 | 고급 분석 필요 시 | ✅ |
# | `SQLAlchemySession` | `agents.extensions.memory` | PostgreSQL, MySQL 등 지원 | 프로덕션 DB | ✅ |
# | `EncryptedSession` | `agents.extensions.memory` | 다른 세션을 암호화로 래핑 | 보안 필요 시 | ✅ |
# | `OpenAIConversationsSession` | `agents` | OpenAI 서버에 대화 저장 | OpenAI 플랫폼 활용 | ✅ |
# | `CustomSession` | 직접 구현 | `Session` 프로토콜 구현 | Django, MongoDB 등 연동 | ✅ |
# | `RedisSession` | `agents.extensions.memory` | Redis 기반 분산 저장 | 대규모 서비스 | ❌ (Redis 서버 필요) |
# | `MongoDBSession` | `agents.extensions.memory` | MongoDB 도큐먼트 저장 | NoSQL 스택 | ❌ (MongoDB 필요) |
# | `DaprSession` | `agents.extensions.memory` | Dapr state store 추상화 | K8s/클라우드 | ❌ (Dapr 사이드카 필요) |
#
# ### 학습 내용
#
# 1. **세션 없이 실행** — 이전 대화 맥락이 유지되지 않는 문제 확인
# 2. **`SQLiteSession`** — 대화 이어가기, 세션 ID별 대화 분리
# 3. **추가 인프라 없이 실행 가능한 6종 실습** — Async/Advanced SQLite, SQLAlchemy(프로덕션 DB),
#    Encrypted(암호화), OpenAIConversations(서버 저장), Custom(직접 구현)
# 4. **실습 문제 - Session으로 챗봇 만들기** — 다중 사용자 챗봇, 개인정보 보호 챗봇,
#    시나리오별 Session 선택 비교

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from agents import Agent, Runner, SQLiteSession

Model = "gpt-5.4-mini"

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
# ---------------------
# ## SDK가 제공하는 다른 Session 유형 실습
#
# 아래 6가지는 **추가 인프라 없이** 바로 실행할 수 있습니다.
# (`RedisSession`, `MongoDBSession`, `DaprSession`은 각각 Redis/MongoDB/Dapr 서버가 필요해 제외)
#
# | 실습 순서 | Session | 시연 포인트 |
# |---|---|---|
# | 1 | `AsyncSQLiteSession` | 비동기 전용 SQLite |
# | 2 | `AdvancedSQLiteSession` | 턴별 토큰 사용량 분석 |
# | 3 | `SQLAlchemySession` | URL만 바꾸면 PostgreSQL/MySQL로 교체 가능 |
# | 4 | `EncryptedSession` | 저장 내용 암호화 (원본 DB에는 암호문만 저장) |
# | 5 | `OpenAIConversationsSession` | 대화를 OpenAI 서버에 저장 |
# | 6 | `CustomSession` | `Session` 프로토콜 직접 구현 |

# %% [markdown]
# ### 1. AsyncSQLiteSession — 비동기 전용 SQLite
#
# `SQLiteSession`과 사용법이 같지만, 내부적으로 `aiosqlite`(비동기 SQLite)를 사용하여
# DB 읽기/쓰기 동안 이벤트 루프를 블로킹하지 않습니다. (03 노트북의 동기/비동기 차이와 같은 원리)

# %%
from agents.extensions.memory import AsyncSQLiteSession

async_session = AsyncSQLiteSession("async_user_1")  # 기본은 인메모리(:memory:)

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=async_session)
print("A1:", result.final_output)

result = await Runner.run(agent, "그 도시는 어느 나라에 있나요?", session=async_session)
print("A2:", result.final_output)  # 맥락 유지 확인

# %% [markdown]
# ### 2. AdvancedSQLiteSession — 브랜치, 토큰 사용량 분석
#
# 기본 세션 기능에 더해 **턴별 토큰 사용량 기록**(`store_run_usage`), **대화 브랜치**
# (`create_branch_from_turn`) 같은 고급 기능을 제공합니다.
# 매 실행 후 `store_run_usage(result)`를 호출하면 세션 전체 사용량을 집계할 수 있습니다.

# %%
from agents.extensions.memory import AdvancedSQLiteSession

adv_session = AdvancedSQLiteSession(session_id="adv_user_1", create_tables=True)

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=adv_session)
await adv_session.store_run_usage(result)   # 이 턴의 토큰 사용량 기록

result = await Runner.run(agent, "그 도시는 어느 나라에 있나요?", session=adv_session)
await adv_session.store_run_usage(result)

print("A2:", result.final_output)
print("세션 전체 사용량:", await adv_session.get_session_usage())

# %% [markdown]
# ### 3. SQLAlchemySession — 프로덕션 DB (PostgreSQL, MySQL 등)
#
# SQLAlchemy가 지원하는 모든 DB를 세션 저장소로 사용할 수 있습니다.
# 여기서는 서버 없이 시연하기 위해 SQLite URL을 사용하지만, **URL만 바꾸면 그대로 프로덕션 DB가 됩니다**.
#
# ```python
# # 프로덕션 예시 (코드는 동일, URL만 교체)
# SQLAlchemySession.from_url("user_1", url="postgresql+asyncpg://user:pw@host/db", create_tables=True)
# SQLAlchemySession.from_url("user_1", url="mysql+aiomysql://user:pw@host/db", create_tables=True)
# ```

# %%
from agents.extensions.memory import SQLAlchemySession

sa_session = SQLAlchemySession.from_url(
    "sa_user_1",
    url="sqlite+aiosqlite:///:memory:",  # 시연용 SQLite (프로덕션은 URL만 교체)
    create_tables=True,
)

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=sa_session)
print("A1:", result.final_output)

result = await Runner.run(agent, "그 도시는 어느 나라에 있나요?", session=sa_session)
print("A2:", result.final_output)  # 맥락 유지 확인

# %% [markdown]
# ### 4. EncryptedSession — 암호화 래퍼
#
# 다른 세션(여기서는 `SQLiteSession`)을 **감싸서** 저장되는 내용을 암호화합니다.
# 원본 세션에서 직접 읽으면 **암호문**만 보이고, `EncryptedSession`을 통해 읽어야 복호화됩니다.
#
# - `encryption_key`: 실제 서비스에서는 환경변수 등으로 안전하게 관리
# - `ttl`: 복호화 유효 시간(초) — 기간이 지난 메시지는 자동으로 무시됨

# %%
from agents.extensions.memory import EncryptedSession

underlying = SQLiteSession("secure_user_1")  # 실제 저장을 담당하는 원본 세션

enc_session = EncryptedSession(
    session_id="secure_user_1",
    underlying_session=underlying,
    encryption_key="my-secret-key",  # 실제 서비스에서는 환경변수로 관리
)

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=enc_session)
print("A1:", result.final_output)

# 원본 세션에서 직접 읽으면 → 암호문만 보임
raw = await underlying.get_items(limit=1)
print("\n[원본 DB에 저장된 내용 - 암호문]")
print(str(raw[0])[:150], "...")

# EncryptedSession을 통해 읽으면 → 복호화된 내용
decrypted = await enc_session.get_items(limit=1)
print("\n[EncryptedSession으로 읽은 내용 - 복호화됨]")
print(str(decrypted[0])[:150], "...")

# %% [markdown]
# ### 5. OpenAIConversationsSession — OpenAI 서버 저장
#
# 대화 기록을 로컬 DB가 아닌 **OpenAI 서버(Conversations API)** 에 저장합니다.
# 별도 인프라 없이 API 키만 있으면 되고, 여러 서버/기기에서 같은 대화를 이어갈 수 있습니다.

# %%
from agents import OpenAIConversationsSession

oai_session = OpenAIConversationsSession()  # 대화가 OpenAI 서버에 저장됨

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=oai_session)
print("A1:", result.final_output)

result = await Runner.run(agent, "그 도시는 어느 나라에 있나요?", session=oai_session)
print("A2:", result.final_output)  # 맥락 유지 확인 (로컬 DB 파일 없음)


# %% [markdown]
# ### 6. CustomSession — Session 프로토콜 직접 구현
#
# `Session` 프로토콜의 4개 메서드(`get_items`, `add_items`, `pop_item`, `clear_session`)만 구현하면
# **어떤 저장소든** 세션으로 쓸 수 있습니다. (Django ORM, MongoDB, 사내 DB 등)
#
# 아래는 가장 단순한 **인메모리 리스트** 구현입니다.

# %%
class MemorySession:
    """Session 프로토콜을 직접 구현한 인메모리 세션"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._items = []   # 빈 리스트로 시작 - 이 세션의 "저장소"

    async def get_items(self, limit=None):
        return list(self._items) if limit is None else list(self._items)[-limit:]

    async def add_items(self, items):
        self._items.extend(items)

    async def pop_item(self):
        return self._items.pop() if self._items else None

    async def clear_session(self):
        self._items.clear()


custom_session = MemorySession("custom_user_1")

result = await Runner.run(agent, "남산 팔각정은 어느 도시에 있나요?", session=custom_session)
print("A1:", result.final_output)

result = await Runner.run(agent, "그 도시는 어느 나라에 있나요?", session=custom_session)
print("A2:", result.final_output)  # 맥락 유지 확인

print("저장된 메시지 개수:", len(custom_session._items))

# %%

# %% [markdown]
# ### 실습 문제 - Session으로 챗봇 만들기
#
# 1. **다중 사용자 챗봇**:
#    여러 사용자가 동시에 사용해도 **대화가 섞이지 않는** 상담 챗봇을 만드세요.
#    - `chat(user_id, message)` 함수를 구현하세요. 내부에서 사용자별로
#      `SQLiteSession(user_id, "chatbot.db")`을 만들어 재사용합니다.
#    - 두 사용자(예: "철수", "영희")가 각각 자기소개(`"저는 부산에 살아요"` / `"저는 제주에 살아요"`)를 한 뒤,
#      번갈아 `"제가 어디 산다고 했죠?"`라고 물었을 때 **각자 자신의 정보로** 답하는지 확인하세요.
#
# 2. **개인정보 보호 챗봇**:
#    1번 챗봇을 **은행 상담 챗봇**으로 업그레이드하세요.
#    고객이 계좌번호 같은 민감 정보를 말해도 **DB 파일에는 평문이 남지 않아야** 합니다.
#    - `SQLiteSession`을 `EncryptedSession`으로 감싸서 구현
#    - 고객: `"제 계좌번호는 123-456-789입니다. 기억해 주세요."` → 다음 턴: `"제 계좌번호가 뭐라고 했죠?"`
#    - 챗봇은 계좌번호를 기억해서 답하지만, **원본 세션(`get_items`)에는 암호문만** 있는 것을 확인하세요.
#
# 3. **(비교/토론 - 코드 작성 없음)**:
#    다음 챗봇 시나리오에 가장 적합한 Session 유형을 고르고, 이유를 한 줄씩 설명해 보세요.
#    - (A) 사내 해커톤에서 시연할 프로토타입 챗봇
#    - (B) 환자 정보를 다루는 병원 예약 챗봇
#    - (C) 서버 여러 대로 운영하는 대규모 쇼핑몰 고객센터 챗봇
#    - (D) 웹·모바일 어디서 접속해도 대화가 이어지는 개인 비서 챗봇
#    - (E) 기존 Django 웹서비스에 추가하는 고객 문의 챗봇
#
#
# ### 테스트 입력 예시
#
# * 1번 👉 철수: "부산에 사신다고 했습니다" / 영희: "제주에 사신다고 했습니다" (대화 분리 확인)
# * 2번 👉 챗봇은 "123-456-789"를 기억해서 답하지만, 원본 세션에는 `'payload': 'gAAAA...'` 암호문만 존재

# %%
