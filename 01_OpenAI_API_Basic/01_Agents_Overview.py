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
# # 01. OpenAI Agent (Agents Overview)
#
# OpenAI 플랫폼을 사용하여 사용자를 대신해 행동(예: 컴퓨터 제어 등)을 수행할 수 있는 **에이전트(Agent)** 를 구축할 수 있습니다.  
# Python용 **Agent SDK**를 사용하면 백엔드에서 이러한 에이전트의 **오케스트레이션(조율) 로직**을 만들 수 있습니다.
#
# **OpenAI Agents SDK**는 **에이전트 기반 AI 애플리케이션**을 개발할 수 있게 해주는 도구입니다.  
#
# ### 주요 기능 요약:
#
# - **에이전트 루프 (Agent Loop)**:  
#   도구 실행 → 결과 전달 → LLM 호출 → 반복 실행 → 완료까지 자동 처리
#
# - **핸드오프 (Handoffs)**:  
#   여러 에이전트 간의 **협업과 위임**을 유연하게 처리 가능
#
# - **가드레일 (Guardrails)**:  
#   에이전트 입력을 **사전 검사/검증**하여, 조건을 만족하지 않으면 **조기 종료 가능**
#
# - **함수 기반 도구 (Function Tools)**:  
#   Python 함수 하나를 **자동으로 에이전트 도구로 변환**,  
#   **Pydantic 기반 스키마 자동 생성** 및 검증 포함
#
# - **추적(Tracing)**:  
#   워크플로우를 **시각화/디버깅/모니터링** 가능하며,  
#   OpenAI의 평가/파인튜닝/디스틸레이션 툴과 통합 가능

# %%
# pip install openai-agents

# %%
from dotenv import load_dotenv
load_dotenv() 

# %%
import openai

Model = "gpt-5-nano"

# %% [markdown]
# ### Hello World 예제
#
# | 메서드                        | 호출 방식                   | 특징                                  |
# | -------------------------- | ----------------------- | ----------------------------------- |
# | `Runner.run(...)`          | `await Runner.run(...)` | 비동기적, 에이전트 루프 자동 실행, 도구 & 핸드오프 지원   |
# | `Runner.run_sync(...)`     | `Runner.run_sync(...)`  | 동기 실행으로 첫 번째 메서드 래핑, 스크립트/테스트 환경 적합 |
# | `Runner.run_streamed(...)` | 비동기 + 스트리밍              | 중간 응답을 이벤트로 실시간 전송 가능             
# - Jupyter notebook은 기본적으로 이벤트 루프가 이미 실행 중이므로  `await Runner.run(...)` 사용  |
#

# %%
# Agent(에이전트 정의)와 Runner(실행 관리자) 불러오기
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="당신은 도움되는 도우미입니다.",
    model=Model
)

# 비동기적으로 에이전트를 실행하여 사용자 요청에 대한 응답을 받음
# 요청: "재귀적 프로그래밍에 대한 짧은 시를 3줄 이내로 써주세요."
result = await Runner.run(starting_agent=agent, 
                          input="재귀적 프로그래밍에 대한 짧은 시를 3줄 이내로 써주세요.")

# 최종 응답 결과를 출력
print(result.final_output)

# %% [markdown]
# ### Simple Handoff Example
#
# 언어에 따라 적절한 에이전트에 작업을 위임(handoff)합니다.
#
# Handoffs는 LLM에게 **도구(tool)** 로 표현됩니다.  
# 예) `Korean agent`에 대한 핸드오프 → LLM 도구 이름: `transfer_to_korean_agent`
#
# **핸드오프 지정 방법 2가지:**
# - **Agent 인스턴스 직접 전달** : `handoffs=[korean_agent, english_agent]`
# - **`handoff()` 함수 사용** : `handoffs=[handoff(agent, on_handoff=콜백, ...)]`  
#   → 콜백(`on_handoff`), 도구 이름/설명 재정의, 입력 데이터 타입, 입력 필터 등 **고급 옵션** 제공  
#   → 심화 내용은 `02_Handoffs.py` 참고

# %%
from agents import Agent, Runner

# 한국어 에이전트 생성: 한국어만 사용 가능
korean_agent = Agent(
    name="Korean agent",
    instructions="당신은 한국어만 할 수 있습니다.",
    model=Model
)

# 영어 에이전트 생성: 영어만 사용 가능
english_agent = Agent(
    name="English agent",
    instructions="당신은 영어만 할 수 있습니다.",
    model=Model
)

# 분류 역할의 핸드오프 에이전트 생성
# 입력된 문장의 언어를 판별하여 적절한 에이전트(한국어 or 영어)에게 전달
handoff_agent = Agent(
    name="Classify agent",
    instructions="요청에 사용된 언어에 따라 적절한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[korean_agent, english_agent],  # 연결할 하위 에이전트 목록
)

# Agent orchenstration 실행
result = await Runner.run(handoff_agent, input="폭싹 속았쑤까?")
print(result.final_output)  # 한국어 에이전트가 응답

result = await Runner.run(handoff_agent, input="Are you happy?")
print(result.final_output)  # 영어 에이전트가 응답

# %% [markdown]
# #### `handoff()` 함수를 사용한 예시
#
# `handoff()` 함수를 사용하면 Agent 직접 전달과 동일하게 동작하지만,  
# `tool_name_override`, `tool_description_override`, `on_handoff` 등 **추가 옵션**을 지정할 수 있습니다.

# %%
from agents import handoff

# Triage Agent 정의 — 사용자 요청의 언어를 판단하여 적절한 에이전트로 위임
triage_agent = Agent(
    name="Triage agent",
    instructions="요청에 사용된 언어에 따라 적절한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[
        # 방법 1: Agent 인스턴스 직접 전달
        # → 자동으로 "transfer_to_korean_agent"라는 도구가 생성됨
        korean_agent,

        # 방법 2: handoff() 함수 사용 — 도구 이름과 설명을 직접 지정 가능
        # → tool_name_override: LLM이 호출할 도구 이름을 커스터마이징
        # → tool_description_override: LLM이 이 핸드오프를 선택하는 판단 기준 설명
        handoff(
            english_agent,
            tool_name_override="transfer_to_english",        # 기본값 대신 커스텀 도구 이름
            tool_description_override="영어 요청일 때 사용",   # LLM에게 전달되는 도구 설명
        ),
    ],
)

# 영어로 질문 → Triage Agent가 영어 요청으로 판단 → english_agent로 핸드오프
result = await Runner.run(triage_agent, input="What is the capital of France?")

# english_agent가 생성한 최종 응답 출력
print(result.final_output)

# %% [markdown]
# ------------------------
# ## 도구 (Tools)
#
# **도구(Tools)** 는 에이전트가 **행동을 수행할 수 있도록 해주는 기능**입니다.  
# 예를 들어, **데이터를 가져오거나**, **코드를 실행하고**, **외부 API를 호출하거나**, 심지어 **컴퓨터를 사용하는 작업**까지도 가능합니다.
#
# Agent SDK에서는 도구를 다음 **세 가지 유형**으로 구분합니다:  
#
# ### 1. **Hosted tools (호스팅 도구)**  
# 이 도구들은 **LLM 서버 내에서**, AI 모델과 함께 실행됩니다.  
# OpenAI는 다음과 같은 호스팅 도구를 제공합니다:
# - 검색 기반 정보 검색 (Retrieval)
# - 웹 검색 (Web search)
# - 컴퓨터 사용 (Computer use)
#
# ### 2. **Function calling (함수 호출 도구)**  
# 이 방식은 **Python 함수**를 도구로 등록하여 사용할 수 있게 해줍니다. 즉, 여러분이 직접 정의한 함수도 LLM이 도구처럼 호출할 수 있습니다.  
#
# **Agents SDK**에서는 **모든 Python 함수를 도구(tool)** 로 사용할 수 있습니다. SDK는 이러한 도구를 자동으로 설정해줍니다:
#
# 1. **도구 이름**은 Python 함수의 이름에서 자동으로 생성됩니다.   
# 2. **도구 설명(description)** 은 함수의 **docstring**에서 가져옵니다.   
# 3. 함수 입력값의 **스키마(schema)** 는 함수의 **인자(arguments)** 로부터 자동 생성됩니다. 
# 4. 각 입력값에 대한 **설명**도 함수의 docstring에서 추출됩니다.  

# %% [markdown]
# `Runner.run()`을 호출하면, 최종 결과가 나올 때까지 루프가 실행됩니다.  이 루프의 동작 방식은 다음과 같습니다:
#
# 1. **LLM 호출:**  
#    에이전트에 설정된 모델과 옵션, 그리고 대화 기록(message history)을 바탕으로 LLM을 호출.  
#
# 2. **LLM 응답 처리:**  
#    모델이 응답을 반환하며, 여기에는 **툴 호출(tool calls)**이 포함될 수 있습니다.
#
# 3. **최종 출력이 있는 경우:**  
#    응답에 **최종 출력(final output)** 이 포함되어 있다면, 이를 반환하고 루프를 종료.
#
# 4. **핸드오프가 있는 경우:**  
#    응답에 **다른 에이전트로의 핸드오프(handoff)** 가 있으면, 현재 에이전트를 새로운 에이전트로 설정한 후 1단계부터 다시 시작.
#
# 5. **툴 호출 처리:**  
#    툴 호출이 있을 경우, 해당 툴을 실행하고 결과 메시지를 기록한 다음 1단계로 되돌아갑니다.
#
#
# **Final Output(최종 출력)** 은 루프에서 에이전트가 생성한 마지막 결과물입니다.
#
# - 에이전트에 `output_type`이 설정되어 있다면:  
#   → LLM이 해당 형식에 맞는 structured output(구조화된 출력)을 반환할 때 루프가 종료
#
# - `output_type`이 설정되지 않은 경우:  
#   → 툴 호출이나 핸드오프가 포함되지 않은 첫 번째 LLM 응답이 최종 출력으로 간주

# %%
# 함수 호출 example
from agents import function_tool
import requests

@function_tool
def multiply(x: float, y: float) -> float:
    """
    x 와 y 를 곱한다.
    """
    print("** multiply 함수 실행 **", x, y)
    return x * y

@function_tool
def get_weather(위도: float, 경도: float) -> str:
    print(위도, 경도)
    print(f"Weather 함수 실행 - 도시: {위도, 경도}")
    latitude=위도
    longitude=경도
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m")
    data = response.json()
    return data['current']['temperature_2m']

agent = Agent(
    name="Assistant",
    instructions=(
        "당신은 유용한 도우미 입니다. 가능하면 제공된 도구를 사용하세요."
        "당신은 기상 전문가 입니다. 도시의 경도와 위도를 주면 날씨를 알려줍니다."
        "사용자 질의에 답하기 위해 당신의 지식에만 너무 의존하지 말고 도구를 사용하세요."
    ),
    model="gpt-5-nano",
    tools=[multiply, get_weather]
)

weather_agent = Agent(
    name="Weather Agent",
    instructions="당신은 기상 전문가 입니다. 도시의 경도와 위도를 주면 날씨를 알려줍니다.",
    model="gpt-5-mini",
    tools=[get_weather]
)

result = await Runner.run(agent, input="7.814 곱하기 103.892의 값은?")
print(result.final_output)

# %%
result = await Runner.run(weather_agent, input="서울의 날씨는 어떻습니까?")
print(result.final_output)

# %%
result = await Runner.run(agent, input="서울의 날씨 곱하기 20은 얼마입니까?")
print(result.final_output)

# %% [markdown]
# ### Agents as tools (에이전트를 도구처럼 사용)
#
# `agent.as_tool()`을 사용하면 에이전트를 일반 함수 도구처럼 등록할 수 있습니다.  
# **핸드오프(Handoff)** 와의 핵심 차이:
#
# | 구분 | Handoff | Agents as tools |
# |------|---------|-----------------|
# | 제어권 | 다음 에이전트로 넘어감 | 원래 에이전트가 유지 |
# | 응답 주체 | 핸드오프 받은 에이전트 | 원래 에이전트 |
# | 용도 | 작업 전체를 위임 | 서브 에이전트를 도구처럼 호출 후 결과 활용 |

# %%
from agents import Agent, Runner

summarizer_agent = Agent(
    name="Summarizer",
    instructions="Summarize the given text into one concise sentence. Always respond in the same language as the input text.",
    model=Model
)

translator_agent = Agent(
    name="Translator",
    instructions="주어진 텍스트를 한국어로 번역하세요.",
    model=Model
)

assistant_agent = Agent(
    name="Assistant",
    instructions=(
        "사용자의 요청에 따라 요약 또는 번역 도구를 사용하여 작업을 처리하세요. "
        "두 작업이 모두 필요하면 순서대로 도구를 호출하세요."
    ),
    model=Model,
    tools=[
        summarizer_agent.as_tool(
            tool_name="summarize",
            tool_description="텍스트를 한 문장으로 요약할 때 사용",
        ),
        translator_agent.as_tool(
            tool_name="translate_to_korean",
            tool_description="텍스트를 한국어로 번역할 때 사용",
        ),
    ],
)

text = (
    "The James Webb Space Telescope has captured stunning images of distant galaxies, "
    "revealing details about the early universe that were previously impossible to observe."
)

result = await Runner.run(
    assistant_agent,
    input=f"다음 텍스트를 요약하고 한국어로 번역해주세요:\n\n{text}"
)
print(result.final_output)

# %% [markdown]
# ## Guardrails (가드레일)
#
# **가드레일(Guardrails)** 은 에이전트와 **병렬로 실행되며**, 사용자 입력에 대해 **검사와 유효성 검증**을 수행할 수 있게 해줍니다.
#
# 예를 들어, 아주 똑똑하지만 **느리고 비용이 많이 드는 모델**을 사용하는 에이전트를 악의적인 사용자가 수학 숙제를 대신 풀어달라고 요청하는 경우,
# 빠르고 저렴한 모델을 활용한 가드레일을 실행하여, 사용자의 입력이 악의적인 목적(예: 숙제 대행)인지 먼저 검사할 수 있습니다.  
# 가드레일이 의심스러운 입력을 감지하면 즉시 오류를 발생시켜 고비용 모델의 실행을 막고 시간과 비용을 절약할 수 있습니다.

# %%
from agents import (
    Agent,
    GuardrailFunctionOutput,
    Runner,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)
from pydantic import BaseModel

Model = "o3-mini"

# guardrail 에이전트의 출력 형식을 정의
# is_homework: 숙제인지 여부
# reasoning: 숙제로 판단한 이유 설명
class HomeworkOutput(BaseModel):
    is_homework: bool          # 이 입력이 숙제 질문인지 여부
    reasoning: str             # 판단 이유 설명


# %%
# 숙제 여부를 판단하는 guardrail 전용 에이전트 정의
guardrail_agent = Agent(
    name="Guardrail check",  # 에이전트 이름
    instructions="사용자가 숙제 질문을 하고 있는지 확인하세요.",  # LLM에게 줄 지시문
    model=Model,
    output_type=HomeworkOutput,  # 결과는 HomeworkOutput 형식으로 반환
)

# 수학 질문 전용 튜터 에이전트 정의
math_tutor_agent = Agent(
    name="Math Tutor",  # 에이전트 이름
    handoff_description="수학 질문을 위한 전문 에이전트",  # 다른 에이전트가 handoff할 때 참고하는 설명
    model=Model,
    instructions="당신은 수학 문제를 도와주는 튜터입니다. 각 단계의 이유를 설명하고 예시를 포함하세요.",  # LLM 지시문
)

# 역사 질문 전용 튜터 에이전트 정의
history_tutor_agent = Agent(
    name="History Tutor",  # 에이전트 이름
    handoff_description="역사 질문을 위한 전문 에이전트",  # 다른 에이전트가 handoff할 때 참고하는 설명
    model=Model,
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
#
# 2. **데코레이터 스타일**: 함수 위에 `@input_guardrail` 을 붙이면, 그 함수가 곧바로 `InputGuardrail` 인스턴스가 됨.  
#    - `input_guardrails=[함수]` 처럼 함수만 넘기면 되어 코드가 짧아짐.  
#    - `@input_guardrail(name="...", run_in_parallel=False)` 처럼 데코레이터 인자로 옵션 지정 가능.
#
# 내부적으로 데코레이터는 `InputGuardrail(guardrail_function=...)` 를 만들어 반환하므로, **두 방식은 완전히 동일**하게 동작합니다.  
# 아래는 데코레이터 스타일로 정의한 뒤 `input_guardrails=[homework_guardrail]` 로 등록합니다.

# %%
# 입력이 '숙제 질문'인지 판단하는 guardrail 함수 (데코레이터 스타일)
@input_guardrail
async def homework_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )


# %%
# 사용자 질문이 숙제인지 확인하고,
# 숙제라면 수학 튜터 또는 역사 튜터 에이전트로 전달(handoff)하는 판단 에이전트 정의
handoff_agent = Agent(
    name="Triage Agent",
    instructions="사용자의 숙제 질문을 기반으로 어떤 에이전트를 사용할지 결정하세요.",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[homework_guardrail],
)

# triage_agent를 테스트하는 비동기 함수 정의
async def question(prompt):
    try:
        # 첫 번째 테스트: 역사 관련 숙제 질문
        result = await Runner.run(handoff_agent, prompt)
        print("Output:", result.final_output)  # 출력 결과 출력
    except Exception as e:
        print("Guardrail에 의해 입력이 거부되었습니다:", e)  # guardrail이 작동하면 예외 메시지 출력

# main 함수 실행 (Notebook 또는 async 환경에서 사용 가능)
prompt = "고구려의 첫번째 왕은 누구인가요?"
await question(prompt)

print("-----------------------------------------------------------------------------------------------------------------------------")

prompt = "사과가 상자에 10개가 담겨 있습니다. 5개를 먹고 3개를 추가하면 상자에는 몇개가 남습니까. 단계별로 생각하세요."
await question(prompt)

print("-----------------------------------------------------------------------------------------------------------------------------")

prompt = "사과와 감 중에 어느 것이 더 달아?"
await question(prompt)

# %% [markdown]
# ### 구조화된 출력

# %%
from pydantic import BaseModel
from agents import Agent, Runner

# 캘린더 이벤트 데이터 구조 정의
class CalendarEvent(BaseModel):
    name: str              # 이벤트 이름
    date: str              # 이벤트 날짜
    participants: list[str]  # 참가자 목록

# 에이전트 정의: 텍스트에서 캘린더 이벤트 정보를 추출
agent = Agent(
    name="캘린더 추출기",  # 에이전트 이름
    instructions="텍스트에서 캘린더 이벤트를 추출하세요. "
                 "이벤트 이름, 날짜, 참가자 정보를 구조화된 데이터로 반환하세요.",
    output_type=CalendarEvent,  # 반환 데이터 형식
    model="gpt-5-mini",             # 사용할 LLM 모델
)

# 실행 함수 정의
async def main():
    input_text = (
        "2025년 4월 10일에 '분기 전략 회의'라는 이름의 팀 미팅이 예정되어 있습니다. "
        "참가자는 오길동, 한철수, 김미미입니다."
    )

    result = await Runner.run(agent, input_text)  # 에이전트 실행
    print("추출된 캘린더 이벤트:")
    print(result.final_output)  # 결과 출력

# 비동기 함수 실행
await main()

# %%

# %% [markdown]
# ### 실습 문제
#
# 1. **분류 에이전트**:  
#    사용자 입력이 \*\*‘수학 문제’\*\*인지 \*\*‘기타 질문’\*\*인지 분류하세요.  
#    분류 결과에 따라 적절한 에이전트로 handoff 하도록 하세요.
#
# 3. **수학 에이전트**:  
#    수학 문제일 경우, Python 함수 `calculate_area(length: float, width: float)`를 도구로 등록하여 `직사각형의 넓이`를 계산해주는 역할을 수행하세요.
#
# 4. **일반 에이전트**:  
#    기타 질문에 대해서는 "질문을 이해했지만 수학 관련 질문만 도와드릴 수 있어요." 라고 응답하세요.
#
# 5. **가드레일**:  
#    입력 내용에 \*\*금지어(`해킹`, `폭탄`)\*\*가 포함되어 있으면 guardrail을 작동시켜 에이전트 실행을 **중단**시키세요.  
#    (힌트: InputGuardrail 사용)
#
#
# ### 테스트 입력 예시
#
# * `"가로 5, 세로 7인 직사각형의 넓이를 구해주세요."`  
#   👉 수학 에이전트 → 함수 실행 → 넓이 출력  
#
# * `"어제 뉴스에 나온 해킹 사고에 대해 말해줘"`  
#   👉 가드레일 트리거 → 실행 중단  
#
# * `"오늘 날씨 어때?"`  
#   👉 일반 에이전트 응답

# %%
