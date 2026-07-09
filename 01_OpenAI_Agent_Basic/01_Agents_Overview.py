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
# - **함수 기반 도구 (Function Tools)**:
#   Python 함수 하나를 **자동으로 에이전트 도구로 변환**,  
#   **Pydantic 기반 스키마 자동 생성** 및 검증 포함

# %%
# pip install openai-agents

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
    
#load_dotenv(dotenv_path)

# %%
#import os
#os.environ

# %%
import openai

Model = "gpt-5.4-mini"

# %% [markdown]
# ### Hello World 예제

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
# -------------------------------
# ### Simple Handoff Example
#
# 언어에 따라 적절한 에이전트에 작업을 위임(handoff)합니다.
#
# Handoffs는 LLM에게 **도구(tool)** 로 표현됩니다.  
# 예) `Korean agent`에 대한 핸드오프 → LLM 도구 이름: `transfer_to_korean_agent`
# ```
# 도구 이름 자동 생성 규칙 (에이전트 이름 → 도구 이름 자동 변환)
# "Korean agent"   →  transfer_to_korean_agent
# "Billing agent"  →  transfer_to_billing_agent
# "English agent"  →  transfer_to_english_agent
# ```
#
# **핸드오프 지정 방법 2가지:**
# 1. **Agent 인스턴스 직접 전달** : `handoffs=[korean_agent, english_agent]`
# 2. **`handoff()` 함수 사용** : `handoffs=[handoff(agent, on_handoff=콜백, ...)]`  
#   → 콜백(`on_handoff`), 도구 이름/설명 재정의, 입력 데이터 타입, 입력 필터 등 **고급 옵션** 제공  
#   → 심화 내용은 `04_Handoffs.py` 참고

# %% [markdown]
# #### 1. Agent 인스턴스 직접 전달 예시

# %%
from agents import Agent, Runner

# 한국어 에이전트 생성: 한국어만 사용 가능
korean_agent = Agent(
    name="Korean_agent",
    instructions="당신은 한국어만 할 수 있습니다.",
    model=Model
)

# 영어 에이전트 생성: 영어만 사용 가능
english_agent = Agent(
    name="English_agent",
    instructions="당신은 영어만 할 수 있습니다.",
    model=Model
)

# 분류 역할의 핸드오프 에이전트 생성
# 입력된 문장의 언어를 판별하여 적절한 에이전트(한국어 or 영어)에게 전달
handoff_agent = Agent(
    name="Classify_agent",
    instructions="요청에 사용된 언어에 따라 적절한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[korean_agent, english_agent],  # 연결할 하위 에이전트 목록
)

# Agent orchenstration 실행
result = await Runner.run(handoff_agent, input="당신은 행복합니까?")
print(result.final_output)  # 한국어 에이전트가 응답
print()
result = await Runner.run(handoff_agent, input="Are you happy?")
print(result.last_agent.name)  # 영어 에이전트가 응답
print(result.final_output)  # 영어 에이전트가 응답

# %% [markdown]
# #### 2. `handoff()` 함수를 사용한 예시
#
# `handoff()` 함수를 사용하면 Agent 직접 전달과 동일하게 동작하지만,  
# `tool_name_override`, `tool_description_override`, `on_handoff` 등 **추가 옵션**을 지정할 수 있습니다.
#
# | 옵션 | 기능 | 예시 |
# |---|---|---|
# | `tool_name_override` | 핸드오프 도구의 이름을 커스텀 지정 | `tool_name_override="escalate_to_billing"` |
# | `tool_description_override` | LLM이 언제 이 핸드오프를 선택할지 판단하는 설명 변경 | `tool_description_override="결제 문제 발생시 전달"` |
# | `on_handoff` | 핸드오프 발생 시 실행할 콜백 함수 지정 | 로깅, 알림, 데이터 전달 등 |

# %%
from agents import handoff

# Triage Agent 정의 — 사용자 요청의 언어를 판단하여 적절한 에이전트로 위임
triage_agent = Agent(
    name="Triage_agent",
    instructions="요청에 사용된 언어에 따라 적절한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[
        korean_agent,  # Agent 인스턴스 직접 전달 (transfer_to_korean_agent 자동 생성)
        handoffs(
            english_agent,
            tool_name_override="English_speaking_agent",     # 기본값 대신 커스텀 도구 이름
            tool_description_override="영어로 입력 받았을 때 영어로 답변",   # LLM에게 전달되는 도구 설명
        ),
    ],
)

# 영어로 질문 → Triage Agent가 영어 요청으로 판단 → english_agent로 핸드오프
result = await Runner.run(triage_agent, input="What is the capital of France?")

# english_agent가 생성한 최종 응답 출력
print(result.last_agent.name)  # 영어 에이전트가 응답
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
# **`Runner.run()`** 을 호출하면, 최종 결과가 나올 때까지 루프가 실행됩니다.  이 루프의 동작 방식은 다음과 같습니다:
#
# 1. **LLM 호출:**  
#    에이전트에 설정된 모델과 옵션, 그리고 대화 기록(message history)을 바탕으로 LLM을 호출.  
#
# 2. **LLM 응답 처리:**  
#    모델이 응답을 반환하며, 여기에는 **툴 호출(tool calls)** 이 포함될 수 있습니다.
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
# ```
# Runner.run() 호출
#       ↓
#   LLM 호출 (1) 
#       ↓                     
#   LLM 응답 수신               
#       ↓                      
#   응답 타입 판단               
#   ┌───┴─────────┐──────────┐      
# 최종출력       핸드오프     툴 호출
#   ↓             ↓          ↓
# 결과반환    새에이전트 교체  툴 실행
#   ↓             |           |
# 루프종료         └───────────┘
#                        ↓
#                   다시 LLM 호출 -> (1)
# ```
#
# --------------------------
# ### Final Output
# **Final Output(최종 출력)** 은 루프에서 에이전트가 생성한 마지막 결과물입니다.
#
# - 에이전트에 `output_type`이 설정되어 있다면:  
#   → LLM이 해당 형식에 맞는 structured output(구조화된 출력)을 반환할 때 루프가 종료
#
# - `output_type`이 설정되지 않은 경우:  
#   → 툴 호출이나 핸드오프가 포함되지 않은 첫 번째 LLM 응답이 최종 출력으로 간주

# %%
from agents import Agent, Runner, function_tool
import requests

# ---- 도구 정의 ----
@function_tool
def multiply(x: float, y: float) -> float:
    """x 와 y 를 곱한다."""
    print("** multiply 함수 실행 **", x, y)
    return x * y

@function_tool
def get_weather(latitude: float, longitude: float) -> str:
    """위도와 경도를 받아 현재 기온을 반환한다."""
    print(f"Weather 함수 실행 - 위도: {latitude}, 경도: {longitude}")
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    )
    data = response.json()
    return data['current']['temperature_2m']


# %% [markdown]
# #### Case 1 - output_type 미지정
# - LLM이 자연어 문자열로 직접 응답 

# %%
# Case 1: output_type 없음
agent_no_output_type = Agent(
    name="Assistant",
    instructions="유용한 도우미입니다. 가능하면 제공된 도구를 사용하세요.",
    model=Model,
    tools=[multiply, get_weather]
)

result1 = await Runner.run(agent_no_output_type, input="7.814 곱하기 103.892의 값은?")
result1.final_output

# %% [markdown]
# #### Case 2 - output_type 지정
# - 구조화된 출력

# %%
# Case 2: output_type 있음 
# LLM이 구조화된 형식(structured output)을 반환할 때 루프 종료
from pydantic import BaseModel

class WeatherResult(BaseModel):
    city: str
    latitude: float
    longitude: float
    temperature: float
    unit: str = "°C"

# 구조화된 출력을 반환하는 agent
weather_agent = Agent(
    name="Weather_Agent",
    instructions=(
        "기상 전문가입니다. 도시의 경도와 위도로 날씨를 알려줍니다."
        "반드시 도구를 사용하여 실제 기온을 조회하세요."
    ),
    model=Model,
    tools=[get_weather],
    output_type=WeatherResult  # structured output 지정
)

result2 = await Runner.run(weather_agent, input="서울(위도 37.5665, 경도 126.9780)의 날씨는?")
print(result2.last_agent.name)  
print(result2.final_output)

# %% [markdown]
# #### Case 3 - 핸드오프 + output_type
#
# - 전문 에이전트에 output_type 설정 

# %%
# Case 3: 핸드오프 포함 

# 계산은 직접, 날씨는 전문 agent에게 handoff 하는 agent
triage_agent = Agent(
    name="Assistant",
    instructions=(
        "유용한 도우미입니다."
        "날씨 관련 질문은 Weather Agent에게 핸드오프하세요."
        "계산 관련 질문은 직접 처리하세요."
    ),
    model=Model,
    tools=[multiply],
    handoffs=[weather_agent]  # 핸드오프 연결
)

result3 = await Runner.run(triage_agent, input="서울 날씨 알려줘")
print(result3.last_agent.name)  
print(result3.final_output)
print()

result4 = await Runner.run(triage_agent, input="7.814 곱하기 103.892의 값은?")
print(result4.last_agent.name)  
print(result4.final_output)

# %% [markdown]
# ----------------------------
# #### 세 가지 케이스 흐름 요약
# ```
# Case 1 (output_type 없음)
#   질문 → multiply 툴 실행 → LLM 자연어 답변 → 루프 종료
#
# Case 2 (output_type 있음)
#   질문 → get_weather 툴 실행 → WeatherResult 형식 충족 → 루프 종료
#
# Case 3 (핸드오프)
#   질문 → triage_agent 판단 → weather_agent 핸드오프
#        → get_weather 툴 실행 → WeatherResult 형식 충족 → 루프 종료
# ```

# %% [markdown]
# ------------------------------
# ### Agents as tools (에이전트를 도구처럼 사용)
#
# - `agent.as_tool()`을 사용하면 에이전트를 일반 함수 도구처럼 등록할 수 있습니다.    
# - 여러 하위 agent의 결과를 취합·비교 분석하여 최종 답변을 생성하는 오케스트레이터 패턴에 적합 (Handoff로는 불가능한 구조)
# ```
# 사용 예시:
# - 서울 날씨 + 도쿄 날씨 → 비교 분석
# - 한국어 번역 + 일본어 번역 → 두 결과 동시 제공
# - 여러 DB 조회 → 결과 취합 후 리포트 생성
# ```
#
# **핸드오프(Handoff)** 와의 차이:
#
# | 구분 | Handoff | Agents as tools |
# |------|---------|-----------------|
# | 제어권 | 다음 에이전트로 넘어감 | 원래 에이전트가 유지 |
# | 응답 주체 | 핸드오프 받은 에이전트 | 원래 에이전트 |
# | 용도 | 작업 전체를 위임 | 서브 에이전트를 도구처럼 호출 후 결과 활용 |
# | 핵심 차이 | 작업 전체를 떠넘기고 본인은 끝 | 부하직원 시켜서 결과만 받아오고 본인이 마무리 |

# %%
from agents import Agent, Runner

# 입력 텍스트를 한 문장으로 요약하는 에이전트입니다.
summarizer_agent = Agent(
    name="Summarizer",
    instructions="주어진 텍스트를 하나의 간결한 문장으로 요약하세요. 항상 입력 텍스트와 동일한 언어로 응답하세요.",
    model=Model
)

# 입력 텍스트를 한국어로 번역하는 에이전트
translator_agent = Agent(
    name="Translator",
    instructions="주어진 텍스트를 한국어로 번역하세요.",
    model=Model
)

# 사용자의 요청을 분석하고,
# 요약 에이전트와 번역 에이전트를 도구처럼 호출하는 상위 에이전트
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
    input=f"다음 텍스트를 한국어로 번역하고 요약해 주세요:\n\n{text}"
)

print(result.last_agent.name)  
print(result.final_output)

# %% [markdown]
# ------------------------------------------
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
    name="캘린더_추출기",  # 에이전트 이름
    instructions="텍스트에서 캘린더 이벤트를 추출하세요. "
                 "이벤트 이름, 날짜, 참가자 정보를 구조화된 데이터로 반환하세요.",
    output_type=CalendarEvent,  # 반환 데이터 형식
    model=Model,             # 사용할 LLM 모델
)

# 실행 함수 정의
async def main():
    input_text = (
        "2025년 4월 10일에 '분기 전략 회의'라는 이름의 팀 미팅이 예정되어 있습니다. "
        "참가자는 오길동, 한철수, 김미미입니다."
    )

    result = await Runner.run(agent, input_text)  # 에이전트 실행
    print("추출된 캘린더 이벤트:")
    print(result.last_agent.name) 
    print(result.final_output)  # 결과 출력

# 비동기 함수 실행
await main()

# %%

# %% [markdown]
# ### 실습 문제
#
# 1. **분류 에이전트**:
#    사용자 입력이 ‘수학 문제’인지 ‘기타 질문’인지 분류하세요.
#    분류 결과에 따라 적절한 에이전트로 handoff 하도록 하세요.
#
# 2. **수학 에이전트**:
#    수학 문제일 경우, Python 함수 `calculate_area(length: float, width: float)`를 도구로 등록하여 `직사각형의 넓이`를 계산해주는 역할을 수행하세요.
#
# 3. **일반 에이전트**:
#    기타 질문에 대해서는 "질문을 이해했지만 수학 관련 질문만 도와드릴 수 있어요." 라고 응답하세요.
#
#
# ### 테스트 입력 예시
#
# * `"가로 5, 세로 7인 직사각형의 넓이를 구해주세요."`
#   👉 수학 에이전트 → 함수 실행 → 넓이 출력
#
# * `"오늘 날씨 어때?"`
#   👉 일반 에이전트 응답

# %%
