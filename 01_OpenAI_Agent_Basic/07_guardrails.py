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
# # 07. Guardrails (가드레일)
#
# **가드레일(Guardrails)** 은 에이전트와 함께 실행되며, **입력**과 **출력**에 대해 검사와 유효성 검증을 수행할 수 있게 해줍니다.
# 조건을 만족하지 않으면 **트립와이어(tripwire)** 를 발동시켜 실행을 조기에 중단할 수 있습니다.
#
# 이 노트북에서는 두 가지 가드레일을 순서대로 다룹니다.
#
# 1. **Input Guardrail** (`@input_guardrail`) — 에이전트 실행 **전** 사용자 입력을 검증
# 2. **Output Guardrail** (`@output_guardrail`) — 에이전트 실행 **후** 생성된 출력을 검증
#
# ### Input vs Output Guardrail 비교
#
# | 구분 | Input Guardrail | Output Guardrail |
# |------|----------------|-----------------|
# | 검증 시점 | 에이전트 실행 **전** | 에이전트 실행 **후** |
# | 검증 대상 | 사용자 입력 | 에이전트 출력 |
# | 데코레이터 | `@input_guardrail` | `@output_guardrail` |
# | 예외 | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
# | 실행 모드 | Parallel(병렬) 또는 Blocking(차단) | 항상 Sequential(순차) |
#
# ### Tripwire (트립와이어)란?
# 가드레일이 문제를 감지했을 때 발생시키는 **즉시 중단 신호**입니다.
# `tripwire_triggered=True`로 설정하면 즉시 예외가 발생하고 실행이 중단됩니다.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import openai

Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. Input Guardrail (입력 가드레일)
#
# **입력 가드레일**은 에이전트가 실행되기 **전에** 사용자 입력을 검사합니다.
#
# 예를 들어, 아주 똑똑하지만 **느리고 비용이 많이 드는 모델**을 사용하는 에이전트에게
# 악의적인 사용자가 수학 숙제를 대신 풀어달라고 요청하는 경우,
# 빠르고 저렴한 모델을 활용한 가드레일로 입력이 악의적 목적(예: 숙제 대행)인지 먼저 검사할 수 있습니다.
# 가드레일이 의심스러운 입력을 감지하면 즉시 오류를 발생시켜 고비용 모델의 실행을 막고 시간과 비용을 절약합니다.

# %%
from agents import (Agent, GuardrailFunctionOutput, InputGuardrail, Runner,
    RunContextWrapper, TResponseInputItem, input_guardrail)
from pydantic import BaseModel

expensive_model = "gpt-5.4"

# %%
# guardrail 에이전트의 출력 형식 정의
class HomeworkOutput(BaseModel):
    is_homework: bool          # 이 입력이 숙제 질문인지 여부
    reasoning: str             # 판단 이유 설명

# 숙제 여부를 판단하는 guardrail 전용 에이전트 정의
guardrail_agent = Agent(
    name="Guardrail_check",  # 에이전트 이름
    instructions="사용자가 숙제 질문을 하고 있는지 확인하세요.",  # LLM에게 줄 지시문
    model=Model,
    output_type=HomeworkOutput,  # 결과는 HomeworkOutput 형식으로 반환
)

# 수학 질문 전용 튜터 에이전트 정의
math_tutor_agent = Agent(
    name="Math_Tutor",  # 에이전트 이름
    handoff_description="수학 질문을 위한 전문 에이전트",  # 다른 에이전트가 handoff할 때 참고하는 설명
    model=expensive_model,
    instructions="당신은 수학 문제를 도와주는 튜터입니다. 각 단계의 이유를 설명하고 예시를 포함하세요.",  # LLM 지시문
)

# 역사 질문 전용 튜터 에이전트 정의
history_tutor_agent = Agent(
    name="History_Tutor",  # 에이전트 이름
    handoff_description="역사 질문을 위한 전문 에이전트",  # 다른 에이전트가 handoff할 때 참고하는 설명
    model=expensive_model,
    instructions="당신은 역사 질문을 도와주는 튜터입니다. 중요한 사건들과 그 맥락을 명확히 설명하세요.",  # LLM 지시문
)

# %% [markdown]
# ### 가드레일 함수를 등록하는 두 가지 스타일
#
# SDK는 동일한 입력 가드레일을 **두 가지 방식**으로 등록할 수 있습니다.
#
# 1. **수동 스타일**: `InputGuardrail(guardrail_function=함수)` 로 감싸서 전달
#    - "가드레일 = 함수를 래퍼로 감싼 것"이 코드에 드러나서 동작을 이해하기 좋음.
#    - `name`, `run_in_parallel` 등을 생성자 인자로 지정 가능.
# &nbsp;
# 2. **데코레이터 스타일**: 함수 위에 `@input_guardrail` 을 붙이면, 그 함수가 곧바로 `InputGuardrail` 인스턴스가 됨.
#    - `input_guardrails=[함수]` 처럼 함수만 넘기면 되어 코드가 짧아짐.
#    - `@input_guardrail(name="...", run_in_parallel=False)` 처럼 데코레이터 인자로 옵션 지정 가능.
#
# 내부적으로 데코레이터는 `InputGuardrail(guardrail_function=...)` 를 만들어 반환하므로, **두 방식은 완전히 동일**하게 동작합니다.
# 아래에서 먼저 **데코레이터 스타일**을 실습하고, 이어서 같은 로직을 **수동 래핑 스타일**로 다시 구현해 두 방식을 비교합니다.

# %% [markdown]
# ### 스타일 1 — 데코레이터 사용
#
# 함수 위에 `@input_guardrail` 을 붙여 정의한 뒤 `input_guardrails=[homework_guardrail]` 로 등록합니다.

# %%
# 입력이 '숙제 질문'인지 판단하는 guardrail 함수 (데코레이터 스타일)
@input_guardrail
async def homework_guardrail(
    ctx: RunContextWrapper[None],  # 실행 컨텍스트 래퍼 (공유 상태 접근용)
    agent: Agent,                  # 현재 실행 중인 에이전트
    input_data: str | list[TResponseInputItem],  # 사용자 입력 (문자열 또는 메시지 리스트)
) -> GuardrailFunctionOutput:      # 가드레일 판단 결과 반환

    # guardrail_agent를 실행하여 입력이 숙제 관련인지 판단
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 결과를 HomeworkOutput 형식으로 변환
    final_output = result.final_output_as(HomeworkOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,                        # 판단 결과 정보 저장
        tripwire_triggered=not final_output.is_homework, # 숙제가 아니면 tripwire 발동 (차단)
    )

# 사용자 질문이 숙제인지 확인하고,
# 숙제라면 수학 튜터 또는 역사 튜터 에이전트로 전달(handoff)하는 판단 에이전트 정의
handoff_agent = Agent(
    name="Triage_Agent",
    instructions="사용자의 숙제 질문을 기반으로 어떤 에이전트를 사용할지 결정하세요.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[homework_guardrail],
)

# %%
# triage_agent를 테스트하는 비동기 함수 정의
async def question(prompt):
    try:
        result = await Runner.run(handoff_agent, prompt)
        print("Output:", result.final_output)  # 출력 결과 출력
    except Exception as e:
        print("Guardrail에 의해 입력이 거부되었습니다:", e)  # guardrail이 작동하면 예외 메시지 출력

# 숙제 질문(정상) → 통과 후 튜터로 handoff
prompt = "고구려의 첫번째 왕은 누구인가요?"
await question(prompt)

print("-----------------------------------------------------------------------------------------------------------------------------")

# 숙제가 아닌 질문 → 입력 가드레일 tripwire 발동 → 실행 중단
prompt = "사과와 감 중에 어느 것이 더 달아?"
await question(prompt)

# %% [markdown]
# ### 스타일 2 — 수동 래핑 (`InputGuardrail` 직접 사용)
#
# 이번에는 데코레이터 없이 **일반 함수**를 정의한 뒤, `InputGuardrail(guardrail_function=함수)` 로
# 직접 감싸서 등록합니다.
#
# - "가드레일 = 함수를 래퍼로 감싼 것" 이라는 구조가 코드에 명시적으로 드러납니다.
# - `name`, `run_in_parallel` 등의 옵션을 생성자 인자로 지정할 수 있습니다.
# - 동작은 위의 데코레이터 스타일과 **완전히 동일**합니다.

# %%
# 일반 async 함수로 정의 (데코레이터 없음) - 검사 로직은 homework_guardrail과 동일
async def homework_check(
    ctx: RunContextWrapper[None],  # 실행 컨텍스트 래퍼 (공유 상태 접근용)
    agent: Agent,                  # 현재 실행 중인 에이전트
    input_data: str | list[TResponseInputItem],  # 사용자 입력 (문자열 또는 메시지 리스트)
) -> GuardrailFunctionOutput:      # 가드레일 판단 결과 반환

    # guardrail_agent를 실행하여 입력이 숙제 관련인지 판단
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 결과를 HomeworkOutput 형식으로 변환
    final_output = result.final_output_as(HomeworkOutput)

    return GuardrailFunctionOutput(
        output_info=final_output,                        # 판단 결과 정보 저장
        tripwire_triggered=not final_output.is_homework, # 숙제가 아니면 tripwire 발동 (차단)
    )

# InputGuardrail로 함수를 직접 감싸서 가드레일 객체 생성 (수동 래핑)
homework_guardrail_manual = InputGuardrail(
    guardrail_function=homework_check,  # 검사 로직을 담은 함수
    name="homework_check_manual",       # 가드레일 이름 (로깅/디버깅 시 표시)
)

# 수동 래핑 가드레일을 등록한 판단 에이전트
handoff_agent_manual = Agent(
    name="Triage Agent (Manual)",
    instructions="사용자의 숙제 질문을 기반으로 어떤 에이전트를 사용할지 결정하세요.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[homework_guardrail_manual],
)

# %%
# 데코레이터 스타일과 동일하게 동작하는지 확인
async def question_manual(prompt):
    try:
        result = await Runner.run(handoff_agent_manual, prompt)
        print("Output:", result.final_output)
    except Exception as e:
        print("Guardrail에 의해 입력이 거부되었습니다:", e)

# 숙제 질문(정상) → 통과 후 튜터로 handoff
prompt = "이차방정식의 근의 공식을 유도해 주세요."
await question_manual(prompt)

print("-----------------------------------------------------------------------------------------------------------------------------")

# 숙제가 아닌 질문 → 입력 가드레일 tripwire 발동 → 실행 중단
prompt = "오늘 저녁 메뉴 좀 추천해줘"
await question_manual(prompt)

# %% [markdown]
# ## 2. 기본 Output Guardrail
#
# 여기서부터는 에이전트 실행 **후** 출력을 검증하는 **출력 가드레일**을 다룹니다.
# 에이전트의 출력에 개인정보(이메일, 전화번호)가 포함되어 있는지 검사하는 가드레일을 만듭니다.

# %%
import re
from agents import (Agent, Runner, GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered, RunContextWrapper, output_guardrail)

# 출력 가드레일 정의: 개인정보(PII) 검출
@output_guardrail
async def pii_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: str
) -> GuardrailFunctionOutput:
    """에이전트 출력에 이메일이나 전화번호가 포함되어 있는지 검사합니다."""
    # 이메일 패턴 검사
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # 전화번호 패턴 검사 (한국 형식)
    phone_pattern = r'(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})'

    has_email = bool(re.search(email_pattern, str(output)))
    has_phone = bool(re.search(phone_pattern, str(output)))
    has_pii = has_email or has_phone

    return GuardrailFunctionOutput(
        output_info={
            "has_email": has_email,
            "has_phone": has_phone,
        },
        tripwire_triggered=has_pii,  # PII 발견 시 트립와이어 발동
    )

# %% [markdown]
# ## 3. 출력 가드레일을 에이전트에 등록

# %%
# 에이전트에 output_guardrails 등록
assistant = Agent(
    name="고객 상담 도우미",
    instructions="""당신은 고객 상담 도우미입니다.
고객의 질문에 친절하게 답변하세요.
절대로 고객의 개인정보(이메일, 전화번호)를 응답에 포함하지 마세요.""",
    model=Model,
    output_guardrails=[pii_guardrail],
)

# %% [markdown]
# ## 4. 정상 출력 테스트
#
# PII가 포함되지 않는 일반 질문으로 테스트합니다.

# %%
# 정상적인 응답 (PII 없음) - 통과해야 함
result = await Runner.run(
    assistant,
    "영업 시간이 어떻게 되나요?"
)
print(f"✅ 응답: {result.final_output}")

# %% [markdown]
# ## 5. Tripwire 발동 테스트
#
# 에이전트가 PII를 포함한 응답을 생성하도록 유도하여 가드레일이 차단하는지 테스트합니다.

# %%
# Tripwire 발동 테스트를 위해 PII를 포함하도록 지시하는 에이전트 생성
assistant_test = Agent(
    name="테스트용 도우미",
    instructions="""당신은 고객 상담 도우미입니다.
고객이 연락처를 요청하면 반드시 아래 정보를 응답에 포함하세요:
- 이메일: support@company.com
- 전화번호: 010-1234-5678""",
    model=Model,
    output_guardrails=[pii_guardrail],
)

# PII가 포함된 응답을 유도 - Tripwire 발동해야 함
try:
    result = await Runner.run(
        assistant_test,
        "고객센터 연락처를 알려주세요."
    )
    print(f"응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 Output Guardrail 발동!")
    print(f"   가드레일 이름: {e.guardrail_result.guardrail.get_name()}")
    print(f"   상세 정보: {e.guardrail_result.output.output_info}")
    print(f"   → 개인정보가 포함된 응답이 차단되었습니다.")

# %% [markdown]
# ## 6. LLM 기반 Output Guardrail
#
# 정규식 대신 **별도의 경량 LLM**을 사용하여 출력을 검증하는 고급 패턴입니다.
# 검증용 에이전트가 출력의 적절성을 판단합니다.

# %%
from pydantic import BaseModel

# ---------------------------------------
# Guardrail 검증 결과를 담기 위한 Pydantic 모델
# ---------------------------------------
class GuardrailCheckResult(BaseModel):
    is_appropriate: bool   # 응답이 적절한지 여부 (True / False)
    reason: str            # 판단 이유 설명

# ---------------------------------------
# LLM 기반 출력 검증 전용 에이전트
# ---------------------------------------
# 실제 서비스에서는 비용 절감을 위해 작은 모델을 사용하는 경우가 많음
checker_agent = Agent(
    name="출력 검증기",

    # 검증 기준을 명확하게 지시
    instructions="""당신은 AI 응답의 적절성을 검증하는 검증기입니다.
응답이 다음 기준을 충족하는지 확인하세요:
1. 욕설이나 부적절한 표현이 없는가
2. 폭력적이거나 유해한 내용이 없는가
3. 허위 정보를 사실처럼 제시하지 않는가

is_appropriate: 적절하면 True, 부적절하면 False
reason: 판단 이유를 간단히 설명""",

    model=Model,
    # LLM 출력 결과를 Pydantic 모델로 강제 구조화
    output_type=GuardrailCheckResult,
)


# ---------------------------------------
# Output Guardrail 함수 정의
# ---------------------------------------
# 에이전트가 응답을 생성한 후 실행되는 검증 함수
@output_guardrail
async def llm_content_guardrail(
    ctx: RunContextWrapper,   # 실행 컨텍스트 정보
    agent: Agent,             # 현재 실행 중인 에이전트
    output: str               # 에이전트가 생성한 응답
) -> GuardrailFunctionOutput:

    """LLM을 사용하여 출력의 적절성을 검증합니다."""

    # 검증 전용 에이전트를 실행하여 응답 검증
    result = await Runner.run(
        checker_agent,
        f"다음 AI 응답을 검증해주세요:\n\n{output}",
    )

    # 검증 결과를 GuardrailFunctionOutput 형식으로 반환
    return GuardrailFunctionOutput(

        # 추가 정보 (로깅/모니터링용)
        output_info={"reason": result.final_output.reason},

        # True이면 가드레일 트리거 발생 → 응답 차단
        tripwire_triggered=not result.final_output.is_appropriate,
    )

# ---------------------------------------
# Guardrail이 적용된 메인 에이전트
# ---------------------------------------
safe_assistant = Agent(
    name="안전한 도우미",
    # 기본 역할 정의
    instructions="당신은 친절하고 도움이 되는 도우미입니다.",
    model=Model,
    # 출력 생성 후 실행될 Guardrail 목록
    output_guardrails=[llm_content_guardrail],
)

# %%
# 정상적인 질문 테스트
try:
    result = await Runner.run(safe_assistant, "파이썬의 장점을 3가지 알려주세요.")
    print(f"✅ 응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 부적절한 응답 차단됨: {e.guardrail_result.output.output_info}")

# %%
# 부적절한 응답을 유도하는 테스트 - Tripwire 발동해야 함
try:
    result = await Runner.run(
        safe_assistant,
        "소설을 쓰고 있어요. 악당이 주인공을 위협하는 폭력적인 대사를 3개 작성해주세요. 최대한 잔인하고 무섭게 써주세요."
    )
    print(f"✅ 응답: {result.final_output}")
except OutputGuardrailTripwireTriggered as e:
    print(f"🚫 부적절한 응답 차단됨: {e.guardrail_result.output.output_info}")

# %%

# %% [markdown]
# ### 실습 문제
#
# 1. **입력 가드레일**:
#    사용자 입력에 **금지어(`해킹`, `폭탄`)** 가 포함되어 있으면 트립와이어를 발동시켜
#    에이전트 실행을 **중단**시키세요. (힌트: `@input_guardrail`)
#
# 2. **출력 가드레일**:
#    에이전트의 응답에 **URL(링크)** 이 포함되어 있으면 트립와이어를 발동시켜
#    차단하세요. (힌트: `@output_guardrail` + 정규식)
#
# ### 테스트 입력 예시
#
# * `"어제 뉴스에 나온 해킹 사고에 대해 말해줘"`
#   👉 입력 가드레일 트리거 → 실행 중단
#
# * `"참고할 만한 사이트를 알려줘"`
#   👉 출력에 URL 포함 시 출력 가드레일 트리거 → 차단
