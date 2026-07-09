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
# # 실습 문제 모범답안 - 03. Error Handling
#
# `03_error_handling` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# 1. **폴백 메시지 핸들러**:
#    `max_turns=2` 초과 시 `"요청이 너무 복잡해요. 더 간단히 질문해 주세요."`를 반환하는
#    비동기 핸들러 `on_limit()`을 작성하고, `error_handlers`로 등록하여
#    예외 없이 폴백 메시지가 출력되는지 확인하세요.
#
# 2. **날씨 조회 핸들러 (비동기 I/O)**:
#    1번 핸들러를 수정하여, 폴백 메시지를 반환하기 전에 `httpx.AsyncClient`로
#    5개 도시의 현재 기온을 조회·출력하도록 만드세요.
#    `other_work()` 코루틴과 함께 실행하여 논블로킹임을 확인하세요.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 준비: 턴 초과가 항상 발생하는 에이전트
#
# `ModelSettings(tool_choice="required")` + `reset_tool_choice=False`를 지정하면
# LLM이 **매 턴 반드시 도구를 호출**하므로 최종 답변을 낼 수 없어
# `max_turns` 초과가 항상 발생합니다. (핸들러 실행을 확실히 재현하기 위한 설정)

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

# %% [markdown]
# ## 1. 폴백 메시지 핸들러 (비동기)
#
# `max_turns=2` 초과 시 예외 대신 폴백 메시지를 반환합니다.
# `include_in_history=False`로 폴백 메시지를 대화 기록에 남기지 않습니다.

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

# %% [markdown]
# ## 2. 날씨 조회 핸들러 (비동기 I/O) - 논블로킹 확인
#
# 핸들러가 폴백 메시지를 반환하기 전에 `httpx.AsyncClient`(비동기 HTTP)로
# 5개 도시의 현재 기온을 조회·출력합니다.
#
# `other_work()` 코루틴을 함께 돌려서, 핸들러가 날씨 API 응답을 기다리는 동안에도
# `[다른 코루틴]`이 계속 출력되는 것(논블로킹)을 확인합니다.

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
