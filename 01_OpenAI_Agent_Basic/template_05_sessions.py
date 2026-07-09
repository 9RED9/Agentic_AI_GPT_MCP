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
# 4. **종합 예제: Session으로 챗봇 만들기** — `while True` + `input()` 챗봇 구현,
#    세션 교체만으로 저장 방식이 바뀌는 것 확인
# 5. **실습 문제** — 본문 챗봇을 다른 Session으로 응용 (보안 챗봇, 상담 리포트 챗봇)

# %%
import os
from dotenv import load_dotenv
#load_dotenv() 

# 현재 실행 중인 Python 파일의 디렉토리를 기준으로 .env 경로 설정
#current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd() # 현재경로
parent_dir = os.path.dirname(current_dir) # os.path.dirname()을 한 번 감싸주면 상위 폴더 경로가 됩니다.
dotenv_path = os.path.join(parent_dir, '.env') # 3. 상위 폴더에 있는 진짜 .env 파일 경로 지정

# 디버깅을 위한 출력
print("수정된 .env 예상 경로:", dotenv_path)

# 4. override=True 옵션을 주어 기존의 잘못된 키 값을 확실하게 덮어씁니다.
is_loaded = load_dotenv(dotenv_path, override=True)
print("Env loaded:", is_loaded)

# 5. API 키 확인 (앞뒤 글자만 확인)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"로드된 API Key: {api_key[:12]}...{api_key[-4:]}")
else:
    print("❌ 여전히 API Key를 찾을 수 없습니다. 경로를 다시 확인해주세요.")

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

# %%

# %% [markdown]
# ## SQLiteSession으로 대화 이어가기
#
# 같은 `session` 인스턴스를 넘기면, 이전 턴의 메시지가 자동으로 포함됩니다.

# %%
session = SQLiteSession("conversation_123")
print(session)

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

# %% [markdown]
# ## 세션 ID별로 대화 분리
#
# `session_id`가 다르면 서로 다른 대화로 저장됩니다.
# 예: `SQLiteSession("user_456", "conversations.db")`로 사용자/채널별 분리.

# %%

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

# %% [markdown]
# ### 2. AdvancedSQLiteSession — 브랜치, 토큰 사용량 분석
#
# 기본 세션 기능에 더해 **턴별 토큰 사용량 기록**(`store_run_usage`), **대화 브랜치**
# (`create_branch_from_turn`) 같은 고급 기능을 제공합니다.
# 매 실행 후 `store_run_usage(result)`를 호출하면 세션 전체 사용량을 집계할 수 있습니다.

# %%

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

# %% [markdown]
# ### 4. EncryptedSession — 암호화 래퍼
#
# 다른 세션(여기서는 `SQLiteSession`)을 **감싸서** 저장되는 내용을 암호화합니다.
# 원본 세션에서 직접 읽으면 **암호문**만 보이고, `EncryptedSession`을 통해 읽어야 복호화됩니다.
#
# - `encryption_key`: 실제 서비스에서는 환경변수 등으로 안전하게 관리
# - `ttl`: 복호화 유효 시간(초) — 기간이 지난 메시지는 자동으로 무시됨

# %%
# 원본 세션에서 직접 읽으면 → 암호문만 보임
# EncryptedSession을 통해 읽으면 → 복호화된 내용

# %% [markdown]
# ### 5. OpenAIConversationsSession — OpenAI 서버 저장
#
# 대화 기록을 로컬 DB가 아닌 **OpenAI 서버(Conversations API)** 에 저장합니다.
# 별도 인프라 없이 API 키만 있으면 되고, 여러 서버/기기에서 같은 대화를 이어갈 수 있습니다.

# %%

# %% [markdown]
# ### 6. CustomSession — Session 프로토콜 직접 구현
#
# `Session` 프로토콜의 4개 메서드(`get_items`, `add_items`, `pop_item`, `clear_session`)만 구현하면
# **어떤 저장소든** 세션으로 쓸 수 있습니다. (Django ORM, MongoDB, 사내 DB 등)
#
# 아래는 가장 단순한 **인메모리 리스트** 구현입니다.

# %%
class MemorySession:
    def __init__(self, session_id: str):


# %% [markdown]
# ---------------------
# ## 종합 예제: Session으로 챗봇 만들기
#
# Session의 대표 활용처는 **챗봇**입니다. 구조는 아주 단순합니다.
#
# ```
# while True:
#     input()으로 사용자 입력 받기       ← "종료"면 break
#     Runner.run(agent, 입력, session)   ← 같은 session 반복 사용 → 맥락 자동 유지
#     답변 출력
# ```
#
# - 핵심은 **같은 `session`을 루프 안에서 반복 사용**하는 것입니다. 모델 자체는 기억이 없고,
#   세션이 이전 대화를 붙여주기 때문에 챗봇이 "기억하는 것처럼" 동작합니다.
# - **사용자 ID = 세션 ID**로 하면 사용자별로 대화가 분리됩니다.
# - 아래 `run_chatbot()`은 `session` 인자로 **어떤 Session이든 갈아끼울 수 있게** 만들었습니다.
#   (인자를 생략하면 기본값 `SQLiteSession(user_id, "chatbot.db")` 사용)
#
# > `input()`은 노트북 자동 실행 시 멈추기 때문에, `inputs` 리스트를 주면 **자동 데모**,
# > 주지 않으면 `input()`으로 **직접 대화**하는 겸용 구조로 작성했습니다.

# %%
chatbot = Agent(
    name="Chatbot",
    instructions="친절한 상담 챗봇입니다. 간결하게 답변하세요.",
    model=Model,
)

async def run_chatbot(user_id: str, session=None, inputs: list | None = None):
    """while True 챗봇 루프.

    - session : 사용할 Session (생략 시 SQLiteSession(user_id, "chatbot.db"))
    - inputs=None  : input()으로 직접 대화 (터미널/Jupyter에서 직접 실행)
    - inputs=[...] : 리스트에서 순서대로 입력을 가져오는 자동 데모 (리스트 소진 시 자동 종료)
    """
    if session is None:
        session = SQLiteSession(user_id, "chatbot.db")   # 기본: 사용자 ID = 세션 ID
    scripted = iter(inputs) if inputs is not None else None

    print(f"=== {user_id}님과의 대화 시작 ('종료' 입력 시 끝) ===")
    while True:
        if scripted is None:
            message = input(f"{user_id}: ")              # 직접 입력
        else:
            message = next(scripted, "종료")             # 자동 데모 입력
            print(f"{user_id}: {message}")

        if message.strip() == "종료":
            print("챗봇을 종료합니다.\n")
            break

        result = await Runner.run(chatbot, message, session=session)
        print("챗봇:", result.final_output)


# %% [markdown]
# ### 예제 1: 다중 사용자 챗봇 (SQLiteSession)
#
# - 사용자마다 다른 세션 → **대화가 섞이지 않음**
# - 파일 DB(`chatbot.db`) → `"종료"` 후 **재접속해도 대화가 이어짐**

# %%
# 재실행 시 이전 기록이 누적되지 않도록 초기화 
await SQLiteSession("철수", "chatbot.db").clear_session()
await SQLiteSession("영희", "chatbot.db").clear_session()

# 철수 접속 → 자기소개 → 종료
await run_chatbot("철수", inputs=["저는 부산에 살아요."])

# 영희 접속 → 자기소개 → 종료
await run_chatbot("영희", inputs=["저는 제주에 살아요."])

# 철수 재접속 → 파일 DB 덕분에 이전 대화가 이어짐 (부산이라고 답해야 정상)
await run_chatbot("철수", inputs=["제가 어디 산다고 했죠?"])

# 영희 재접속 → 제주라고 답해야 정상 (철수의 대화와 섞이지 않음)
await run_chatbot("영희", inputs=["제가 어디 산다고 했죠?"])

# %%
# 자유 대화
await run_chatbot("Anyone")

# %% [markdown]
# ### 예제 2: 세션만 교체하면 저장 방식이 바뀐다 (OpenAIConversationsSession)
#
# 챗봇 코드는 **한 줄도 바꾸지 않고**, `session` 인자만 교체하면
# 대화 저장 위치가 로컬 DB → OpenAI 서버로 바뀝니다.
# (실습 문제에서는 이런 식으로 다른 Session을 응용해 봅니다)

# %%
# 세션만 OpenAIConversationsSession으로 교체 → 대화가 OpenAI 서버에 저장됨 (로컬 DB 파일 없음)

# %%

# %% [markdown]
# ### 실습 문제 - 본문 챗봇을 다른 Session으로 응용하기
#
# 본문의 `run_chatbot()`은 이미 완성된 챗봇입니다.
# **세션만 바꿔서** 성격이 다른 챗봇으로 발전시켜 보세요.
#
# 1. **은행 상담 챗봇 (보안 응용 - `EncryptedSession`)**:
#    고객이 계좌번호 같은 민감 정보를 말해도 **DB 파일에는 평문이 남지 않는** 챗봇을 만드세요.
#    - `SQLiteSession("고객_1001", "bank_chatbot.db")`을 `EncryptedSession`으로 감싸서
#      본문 `run_chatbot()`의 `session` 인자로 전달하면 됩니다. (챗봇 코드 수정 불필요)
#    - 대화 예: `"제 계좌번호는 123-456-789입니다. 기억해 주세요."` → `"제 계좌번호가 뭐라고 했죠?"` → `"종료"`
#    - 챗봇은 계좌번호를 기억해서 답하지만, 종료 후 **원본 세션(`get_items`)에는 암호문만**
#      있는 것을 확인하세요.
# >
# 2. **상담 리포트 챗봇 (분석 응용 - `AdvancedSQLiteSession`)**:
#    상담이 끝나면(`"종료"` 입력 시) **이번 상담에 사용된 토큰량을 리포트로 출력**하는 챗봇을 만드세요.
#    - 본문 `run_chatbot()`을 복사하여 `run_chatbot_with_report()`를 만들고 두 곳만 수정합니다:
#      - 세션: `AdvancedSQLiteSession(session_id=user_id, create_tables=True)`
#      - 매 턴 `Runner.run` 후: `await session.store_run_usage(result)` 추가
#      - 루프 종료 후: `await session.get_session_usage()` 출력
#    - 2~3턴 대화 후 종료했을 때 사용량 리포트가 출력되는지 확인하세요.
#
#
# ### 💡 구현 팁
#
# - 1번은 코드를 새로 짤 필요가 없습니다. **세션을 만들어 `session=` 인자로 넘기기만** 하면 됩니다.
#   (본문 예제 2에서 `OpenAIConversationsSession`으로 교체했던 것과 같은 방식)
# - 2번은 루프 안에 한 줄(`store_run_usage`), 루프 뒤에 한 줄(`get_session_usage`)을 추가하는 문제입니다.
#
#
# ### 테스트 입력 예시
#
# * 1번 👉 챗봇은 "123-456-789"를 기억해서 답하지만, 원본 세션에는 `'payload': 'gAAAA...'` 암호문만 존재
# * 2번 👉 종료 시 `{'requests': 2, 'input_tokens': ..., 'total_turns': 2}` 형태의 상담 리포트 출력

# %% [markdown]
# ## 준비: 본문의 챗봇 (`run_chatbot`)
#
# 본문에서 만든 챗봇을 그대로 가져옵니다.
# (실습 확인을 위해 instructions에 "대화에서 알려준 정보는 다시 확인해 준다"는 문구만 추가)

# %%
from agents import Agent, Runner, SQLiteSession

Model = "gpt-5.4-mini"

chatbot = Agent(
    name="Chatbot",
    instructions=(
        "친절한 상담 챗봇 실습용 시뮬레이션입니다. 간결하게 답변하세요. "
        "사용자가 이 대화에서 알려준 정보(계좌번호 등)를 물어보면 대화 기록을 참고해 다시 확인해 주세요."
    ),
    model=Model,
)

async def run_chatbot(user_id: str, session=None, inputs: list | None = None):
    """본문과 동일한 while True 챗봇 루프.

    - session : 사용할 Session (생략 시 SQLiteSession(user_id, "chatbot.db"))
    - inputs=None  : input()으로 직접 대화 / inputs=[...] : 자동 데모
    """
    if session is None:
        session = SQLiteSession(user_id, "chatbot.db")
    scripted = iter(inputs) if inputs is not None else None
    print(type(scripted))
    print(scripted)

    print(f"=== {user_id}님과의 대화 시작 ('종료' 입력 시 끝) ===")
    while True:
        if scripted is None:
            message = input(f"{user_id}: ")
        else:
            message = next(scripted, "종료")
            print(f"{user_id}: {message}")

        print("*** : ", message)
        if message.strip() == "종료":
            print("챗봇을 종료합니다.\n")
            break

        result = await Runner.run(chatbot, message, session=session)
        print("챗봇:", result.final_output)


# %% [markdown]
# ## 1. 은행 상담 챗봇 (보안 응용 - EncryptedSession)
#
# 챗봇 코드는 **한 줄도 수정하지 않습니다.**
# `EncryptedSession`을 만들어 `session` 인자로 넘기기만 하면 됩니다.
# - 챗봇 동작은 동일: 같은 세션을 반복 사용하므로 계좌번호를 기억해서 답변
# - 저장 방식만 변경: DB 파일(`bank_chatbot.db`)에는 **암호문만** 남음
#   → DB 파일이 유출되어도 대화 내용(계좌번호)을 읽을 수 없음

# %%
from agents.extensions.memory import EncryptedSession

underlying = SQLiteSession("고객_1001", "bank_chatbot.db")  # 실제 저장 담당 (파일 DB)

secure_session = EncryptedSession(
    session_id="고객_1001",
    underlying_session=underlying,
    encryption_key="bank-secret-key",  # 실제 서비스에서는 환경변수로 관리
)

await secure_session.clear_session()  # 재실행 시 기록 누적 방지 (모범답안 재현용)

# 본문 챗봇에 세션만 교체해서 전달
await run_chatbot("고객_1001", session=secure_session, inputs=[
    "제 계좌번호는 123-456-789입니다. 기억해 주세요.",
    "제 계좌번호가 뭐라고 했죠?",   # 세션 덕분에 챗봇이 기억해서 답변
])
