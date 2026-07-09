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
# # 실습 문제 모범답안 - 05. Sessions (본문 챗봇을 다른 Session으로 응용)
#
# `05_sessions` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# 본문의 `run_chatbot()`은 이미 완성된 챗봇입니다. **세션만 바꿔서** 발전시키세요.
#
# 1. **은행 상담 챗봇 (보안 응용 - `EncryptedSession`)**:
#    `SQLiteSession`을 `EncryptedSession`으로 감싸 `session` 인자로 전달.
#    챗봇은 계좌번호를 기억하지만, DB 파일에는 암호문만 남는 것을 확인하세요.
#
# 2. **상담 리포트 챗봇 (분석 응용 - `AdvancedSQLiteSession`)**:
#    `"종료"` 시 이번 상담에 사용된 토큰량 리포트를 출력하는 챗봇을 만드세요.

# %%
from dotenv import load_dotenv
load_dotenv()

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

    print(f"=== {user_id}님과의 대화 시작 ('종료' 입력 시 끝) ===")
    while True:
        if scripted is None:
            message = input(f"{user_id}: ")
        else:
            message = next(scripted, "종료")
            print(f"{user_id}: {message}")

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

# %%
# 종료 후 DB 파일에 실제로 저장된 내용 확인 → 암호문만 존재 (계좌번호 평문 없음)
raw = await underlying.get_items(limit=1)
print("[bank_chatbot.db에 저장된 내용 - 암호문]")
print(str(raw[0])[:150], "...")

# 평문 계좌번호가 저장 내용에 포함되어 있지 않은지 검증
all_raw = str(await underlying.get_items())
print("\n원본 DB에 '123-456-789' 평문 포함 여부:", "123-456-789" in all_raw)

# %% [markdown]
# ## 2. 상담 리포트 챗봇 (분석 응용 - AdvancedSQLiteSession)
#
# 본문 `run_chatbot()`을 복사해서 **세 곳만** 수정합니다.
# 1. 세션: `AdvancedSQLiteSession(session_id=user_id, create_tables=True)`
# 2. 매 턴 `Runner.run` 후: `await session.store_run_usage(result)` — 턴별 사용량 기록
# 3. 루프 종료 후: `await session.get_session_usage()` — 상담 리포트 출력

# %%
from agents.extensions.memory import AdvancedSQLiteSession

async def run_chatbot_with_report(user_id: str, inputs: list | None = None):
    """종료 시 이번 상담의 토큰 사용량 리포트를 출력하는 챗봇"""
    session = AdvancedSQLiteSession(session_id=user_id, create_tables=True)  # (수정 1)
    scripted = iter(inputs) if inputs is not None else None

    print(f"=== {user_id}님과의 대화 시작 ('종료' 입력 시 끝) ===")
    while True:
        if scripted is None:
            message = input(f"{user_id}: ")
        else:
            message = next(scripted, "종료")
            print(f"{user_id}: {message}")

        if message.strip() == "종료":
            print("챗봇을 종료합니다.")
            break

        result = await Runner.run(chatbot, message, session=session)
        await session.store_run_usage(result)   # (수정 2) 이 턴의 사용량 기록
        print("챗봇:", result.final_output)

    # (수정 3) 상담 종료 리포트
    print("\n[상담 리포트 - 토큰 사용량]")
    print(await session.get_session_usage())

# %%
await run_chatbot_with_report("고객_2002", inputs=[
    "남산 팔각정은 어느 도시에 있나요?",
    "그 도시는 어느 나라에 있나요?",
])

# %%
await run_chatbot_with_report("고객_2003")

# %% [markdown]
# ### 정리
#
# | 문제 | 핵심 포인트 |
# |------|------|
# | 1. 은행 상담 챗봇 | 챗봇 코드 수정 없이 **세션만 `EncryptedSession`으로 교체** → 챗봇은 기억하지만 DB에는 암호문만 저장 |
# | 2. 상담 리포트 챗봇 | 세션을 `AdvancedSQLiteSession`으로 교체 + 턴마다 `store_run_usage`, 종료 시 `get_session_usage` |
#
# **이번 실습의 교훈**: 챗봇의 대화 로직(`while True` 루프)과 저장 방식(Session)이 분리되어 있어서,
# **세션만 갈아끼우면** 보안 챗봇, 분석 챗봇, 서버 저장 챗봇 등으로 자유롭게 발전시킬 수 있습니다.
