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
# # 03. Error Handling
#
# 에이전트 루프에서 **max_turns**를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생하여 프로그램이 중단됩니다.  
# **error_handlers**를 사용하면 예외 대신 **제어된 최종 메시지**를 반환하도록 할 수 있습니다.
#
# ### 핵심 개념
#
# - **`error_handlers={"max_turns": handler}`**: `Runner.run()` 호출 시 전달하여, 턴 초과 시 핸들러 함수가 실행되도록 설정
# - **`RunErrorHandlerResult`**: 핸들러가 반환하는 결과 객체로, `final_output`(폴백 메시지)과 `include_in_history` 옵션을 포함
# - **`include_in_history=False`**: 폴백 메시지를 대화 기록에 남기지 않음 (사용자 안내 전용)
# - **동기(`def`) vs 비동기(`async def`) 핸들러**: 동기 핸들러는 실행 중 이벤트 루프를 블로킹하고, 비동기 핸들러는 `await` 시 제어권을 양보 → I/O 작업에는 비동기 권장
#
# ### 학습 내용
#
# 1. `max_turns` 초과 시 **기본 동작** 확인: `MaxTurnsExceeded` 예외 발생 → 프로그램 중단
# 2. **같은 코드에 `error_handlers`만 추가**: 예외 없이 폴백 메시지로 안전하게 처리
# 3. `include_in_history` 옵션의 역할과 활용
# 4. 동기(`requests`) vs 비동기(`httpx`) 핸들러의 **이벤트 루프 블로킹 차이**를 실제 날씨 API 호출로 관찰

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
    
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 턴 초과가 항상 발생하는 에이전트 정의
#
# 실험을 위해 **매 턴 반드시 도구를 호출하도록 강제**하여, max_turns 초과가 항상 발생하는 에이전트를 만듭니다.
#
# - `ModelSettings(tool_choice="required")`: LLM이 매 턴 반드시 도구를 호출 → 최종 답변을 낼 수 없음
# - `reset_tool_choice=False`: 첫 도구 호출 후 `tool_choice`가 `"auto"`로 자동 리셋되는 것을 방지
#
# (이 두 설정이 없으면 모델이 도중에 최종 답변을 내버려 max_turns 초과가 발생하지 않을 수 있음)

# %%
from agents import (
    Agent, Runner, function_tool, ModelSettings, MaxTurnsExceeded,
    RunErrorHandlerInput, RunErrorHandlerResult,
)

@function_tool
def NumberTool(n: int) -> int:
    return n

# tool_choice="required" + reset_tool_choice=False
# → 매 턴 반드시 도구를 호출하므로 최종 답변을 낼 수 없어 max_turns 초과가 항상 발생
agent = Agent(
    name="Assistant",
    instructions="NumberTool을 사용해 1부터 5까지 순서대로 세세요.",
    model=Model,
    tools=[NumberTool],
    model_settings=ModelSettings(tool_choice="required"),
    reset_tool_choice=False,
)

# %% [markdown]
# ## 1. 기본 동작: MaxTurnsExceeded 예외 발생
#
# `error_handlers` 없이 실행하면 max_turns 초과 시 **예외가 발생하여 프로그램이 중단**됩니다.
# try/except로 잡지 않으면 사용자는 에러 화면을 보게 됩니다.

# %%
# error_handlers 없이 실행 → MaxTurnsExceeded 예외 발생
try:
    result = await Runner.run(
        agent,
        "1부터 5까지 세어주세요.",
        max_turns=2,
    )
    print(result.final_output)
except MaxTurnsExceeded as e:
    print("예외 발생:", e)


# %% [markdown]
# ## 2. 해결: error_handlers로 exception 없이 처리
#
# **핸들러 함수**를 정의합니다. `RunErrorHandlerInput`을 인자로 받고, `RunErrorHandlerResult`를 반환합니다.
#
# - `final_output`: 예외 대신 사용자에게 돌려줄 **폴백 메시지**
# - `include_in_history=False`: 이 메시지를 대화 히스토리에 넣지 않음 (사용자 안내만 할 때 유용)
# - 핸들러는 `async def`(권장) 또는 `def` 둘 다 가능 — 차이는 아래 "동기 vs 비동기 핸들러" 섹션에서 확인

# %%
# max_turns 초과 시 실행될 핸들러 (예외 대신 폴백 메시지 반환)
async def on_max_turns(_data: RunErrorHandlerInput[None]) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="죄송합니다. 처리 한도를 초과했습니다. 잠시 후 다시 시도해주세요.",
        include_in_history=False,  # 대화 기록에 저장 안 함
    )

# %% [markdown]
# ### error_handlers를 등록하고 실행
#
# 아래 코드는 **1번의 예외 발생 코드와 완전히 동일**하고, `error_handlers` 한 줄만 추가되었습니다.
#
# 이제 max_turns를 초과해도 예외가 나지 않고, 핸들러가 정한 폴백 메시지가 `result.final_output`으로 반환됩니다. (try/except도 필요 없음)

# %%
# 1번과 같은 코드 + error_handlers 한 줄 추가 → 예외 대신 폴백 메시지 반환
result = await Runner.run(
    agent,
    "1부터 5까지 세어주세요.",
    max_turns=2,
    error_handlers={"max_turns": on_max_turns},
)
print(result.final_output)

# %% [markdown]
# ---------------------
# ## 3. 동기 vs 비동기 핸들러
#
# 핸들러는 **동기(`def`)** 와 **비동기(`async def`)** 두 가지 방식으로 작성할 수 있습니다.
# 문법은 비슷하지만 **이벤트 루프에 미치는 영향**이 완전히 다릅니다.
#
# | 구분 | 동기 핸들러 (`def`) | 비동기 핸들러 (`async def`) |
# |---|---|---|
# | 이벤트 루프 | 실행 중 **블로킹** | `await` 시 제어권 **양보** (논블로킹) |
# | 다른 코루틴 | 실행 불가 (대기) | 동시 실행 가능 |
# | 적합한 작업 | 단순 문자열 반환 정도 | DB 조회, HTTP 요청, 로깅 등 I/O 작업 |
# | 실무 권장 | - | ✅ 권장 |
#
# 아래 실험에서는 턴 초과 시 핸들러가 **실제 외부 날씨 API(open-meteo)를 호출**하여
# 5개 도시의 기온을 조회합니다. 같은 작업을 두 가지 방식으로 구현합니다.
#
# - **동기 핸들러**: `requests.get()` — 블로킹 HTTP. 응답을 기다리는 동안 **이벤트 루프 전체가 멈춤**
# - **비동기 핸들러**: `await httpx.AsyncClient().get()` — 논블로킹 HTTP. 응답을 기다리는 동안 **다른 코루틴이 계속 실행**

# %%
import asyncio
import requests   # 동기(블로킹) HTTP 클라이언트
import httpx      # 비동기(논블로킹) HTTP 클라이언트

# 날씨 조회 대상 도시 (위도, 경도)
CITIES = {"서울": (37.57, 126.98), "도쿄": (35.68, 139.69), "뉴욕": (40.71, -74.01),
          "런던": (51.51, -0.13), "파리": (48.85, 2.35)}

def weather_url(lat, lon):
    return f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"

# 두 핸들러가 공통으로 반환할 폴백 메시지
FALLBACK = RunErrorHandlerResult(
    final_output="처리 한도를 초과했습니다. (날씨 로깅 완료)",
    include_in_history=False,
)


# 동기 핸들러: requests가 응답을 기다리는 동안 이벤트 루프 전체가 멈춤 (블로킹)
def sync_weather_handler(_data):
    print(">> [핸들러] 시작 (requests, 동기)")
    for city, (lat, lon) in CITIES.items():
        temp = requests.get(weather_url(lat, lon)).json()["current"]["temperature_2m"]
        print(f">> [핸들러] {city} {temp}°C")
    print(">> [핸들러] 완료")
    return FALLBACK


# 비동기 핸들러: await로 응답을 기다리는 동안 다른 코루틴이 실행됨 (논블로킹)
async def async_weather_handler(_data):
    print(">> [핸들러] 시작 (httpx, 비동기)")
    async with httpx.AsyncClient() as client:
        for city, (lat, lon) in CITIES.items():
            resp = await client.get(weather_url(lat, lon))
            print(f">> [핸들러] {city} {resp.json()['current']['temperature_2m']}°C")
    print(">> [핸들러] 완료")
    return FALLBACK


# %% [markdown]
# ### 블로킹 관찰 실험
#
# 슬라이드의 타임라인을 코드로 재현합니다. 슬라이드의 각 줄이 코드에서 무엇인지 대응시키면:
#
# | 슬라이드 | 코드 |
# |---|---|
# | 에이전트 실행 (max_turns 도달) | `Runner.run(agent, ..., max_turns=2)` |
# | error handler | `sync_weather_handler` / `async_weather_handler` |
# | 다른 코루틴 | `other_work()` — 0.5초마다 한 줄씩 출력 |
#
# **관찰 포인트는 딱 하나입니다:**
# `>> [핸들러] 시작` 과 `>> [핸들러] 완료` **사이에 `[다른 코루틴]` 줄이 출력되는가?**
#
# - 동기 핸들러: 사이에 `[다른 코루틴]` 출력이 **없음** → 날씨 조회 동안 루프가 멈춰 있었다는 증거 (블로킹)
# - 비동기 핸들러: 사이에도 `[다른 코루틴]`이 **계속 출력** → 날씨 조회 동안에도 루프가 돌았다는 증거 (논블로킹)

# %%
# 같은 이벤트 루프에서 함께 돌고 있는 "다른 코루틴" - 0.5초마다 한 줄씩 출력
async def other_work():
    while True:
        await asyncio.sleep(0.5)
        print("   [다른 코루틴] 실행 중")

ticker = asyncio.create_task(other_work())   # 다른 코루틴 시작

result = await Runner.run(                   # 에이전트 실행 (턴 초과 → 핸들러 실행)
    agent, "1부터 5까지 세어주세요.",
    max_turns=2,
    error_handlers={"max_turns": sync_weather_handler},
)

ticker.cancel()                              # 관찰 종료
print("응답:", result.final_output)

# %%
# 2) 비동기 핸들러: [핸들러] 시작~완료 사이에도 [다른 코루틴]이 계속 출력됨 (논블로킹)
ticker = asyncio.create_task(other_work())   # 다른 코루틴 시작

result = await Runner.run(                   # 에이전트 실행 (턴 초과 → 핸들러 실행)
    agent, "1부터 5까지 세어주세요.",
    max_turns=2,
    error_handlers={"max_turns": async_weather_handler},
)

ticker.cancel()                              # 관찰 종료
print("응답:", result.final_output)

# %% [markdown]
# ### 관찰 결과
#
# - **동기 핸들러(requests)**: `[핸들러] 시작`부터 `완료`까지 `[다른 코루틴]` 출력이 **한 줄도 없습니다**.
#   requests가 날씨 API 응답을 기다리는 동안 이벤트 루프 전체가 멈춰 있었기 때문입니다.
#   → 슬라이드 상단(SYNC)의 **"블로킹됨 / 실행 불가(대기)"** 구간이 이것입니다.
#
# - **비동기 핸들러(httpx)**: `[핸들러] 시작`과 `완료` 사이에도 `[다른 코루틴]`이 **계속 출력**됩니다.
#   `await`로 응답을 기다리는 동안 제어권이 이벤트 루프에 반납되어 다른 코루틴이 실행되기 때문입니다.
#   → 슬라이드 하단(ASYNC)의 **"await 양보 / 실행됨!"** 구간이 이것입니다.
#
# **~0초 블로킹인 이유**: async 핸들러는 `await` 시점마다 제어권을 이벤트 루프에 반납하므로,
# 루프를 점유하는 것은 `await` 사이의 짧은 동기 코드 조각(마이크로초~밀리초)뿐입니다.
#
# **실무 결론**: 웹 서버에서 동기 핸들러의 블로킹 구간은 곧 **다른 모든 사용자의 요청이 멈추는 시간**입니다.
# 핸들러에서 DB 조회, HTTP 요청, 로깅 같은 I/O 작업을 한다면 반드시
# `async def` + 비동기 클라이언트(`httpx` 등)를 사용해야 합니다.

# %%

# %% [markdown]
# ### 실습 문제
#
# 1. **폴백 메시지 핸들러**:
#    `max_turns=2` 초과 시 `"요청이 너무 복잡해요. 더 간단히 질문해 주세요."`를 반환하는
#    **비동기 핸들러** `on_limit()`을 작성하고, `error_handlers`로 등록하여
#    예외 없이 폴백 메시지가 출력되는지 확인하세요.
#    (폴백 메시지는 대화 기록에 남기지 마세요)
#
# 2. **날씨 조회 핸들러 (비동기 I/O)**:
#    1번 핸들러를 수정하여, 폴백 메시지를 반환하기 전에 `httpx.AsyncClient`로
#    본문의 `CITIES` 5개 도시의 현재 기온을 조회·출력하도록 만드세요.
#    `other_work()` 코루틴과 함께 실행하여, 핸들러가 날씨를 조회하는 동안에도
#    `[다른 코루틴]`이 계속 출력되는 것(논블로킹)을 확인하세요.
#
#
# ### 테스트 입력 예시
#
# * `"1부터 5까지 세어주세요."` (max_turns=2)
#   * 1번 👉 예외 없이 `"요청이 너무 복잡해요. 더 간단히 질문해 주세요."` 출력
#   * 2번 👉 `[핸들러] 시작 ~ 완료` 사이에도 `[다른 코루틴]` 계속 출력 (논블로킹)

# %%
from agents import (
    Agent, Runner, function_tool, ModelSettings,
    RunErrorHandlerInput, RunErrorHandlerResult,
)

@function_tool
def NumberTool(n: int) -> int:
    """숫자 n을 그대로 반환한다."""
    return n

# 매 턴 반드시 도구를 호출 → max_turns 초과가 항상 발생
agent = Agent(
    name="Assistant",
    instructions="NumberTool을 사용해 1부터 5까지 순서대로 세세요.",
    model=Model,
    tools=[NumberTool],
    model_settings=ModelSettings(tool_choice="required"),
    reset_tool_choice=False,
)

QUESTION = "1부터 5까지 세어주세요."


# %%
async def on_limit(_data):
    return RunErrorHandlerResult(
        final_output="요청이 너무 복잡해요. 더 간단히 질문해 주세요.",
        include_in_history=False,  # 대화 기록에 저장 안 함
    )

result = await Runner.run(
    agent,
    QUESTION,
    max_turns=2,
    error_handlers={"max_turns": on_limit},  # 예외 대신 핸들러 실행
)
print(result.final_output)

# %%
import asyncio
import httpx      # 비동기(논블로킹) HTTP 클라이언트

# 날씨 조회 대상 도시 (위도, 경도)
CITIES = {"서울": (37.57, 126.98), "도쿄": (35.68, 139.69), "뉴욕": (40.71, -74.01),
          "런던": (51.51, -0.13), "파리": (48.85, 2.35)}

def weather_url(lat, lon):
    return f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"


# 비동기 핸들러: await로 응답을 기다리는 동안 다른 코루틴이 실행됨 (논블로킹)
async def on_limit_weather(_data):
    print(">> [핸들러] 시작 (httpx, 비동기)")
    async with httpx.AsyncClient() as client:
        for city, (lat, lon) in CITIES.items():
            resp = await client.get(weather_url(lat, lon))
            print(f">> [핸들러] {city} {resp.json()['current']['temperature_2m']}°C")
    print(">> [핸들러] 완료")
    return RunErrorHandlerResult(
        final_output="요청이 너무 복잡해요. 더 간단히 질문해 주세요.",
        include_in_history=False,
    )


# 같은 이벤트 루프에서 함께 돌고 있는 "다른 코루틴" - 0.5초마다 한 줄씩 출력
async def other_work():
    while True:
        await asyncio.sleep(0.5)
        print("   [다른 코루틴] 실행 중")


# %%
ticker = asyncio.create_task(other_work())   # 다른 코루틴 시작

result = await Runner.run(                   # 에이전트 실행 (턴 초과 → 핸들러 실행)
    agent, QUESTION,
    max_turns=2,
    error_handlers={"max_turns": on_limit_weather},
)

ticker.cancel()                              # 관찰 종료
print("응답:", result.final_output)

# %%
