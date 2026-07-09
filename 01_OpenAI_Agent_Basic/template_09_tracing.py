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
# # 9. Tracing (트레이싱)
#
# 에이전트가 어떤 단계를 거쳐 응답을 생성했는지 **추적(Trace)** 할 수 있습니다.
# 트레이싱은 디버깅, 모니터링, 성능 분석에 필수적인 기능입니다.
#
# ### 핵심 개념
#
# | 개념 | 설명 |
# |------|------|
# | **Trace** | 하나의 워크플로우 전체를 나타내는 단위 |
# | **Span** | Trace 내의 개별 작업 단위 (LLM 호출, 도구 실행 등) |
# | **자동 트레이싱** | SDK가 자동으로 에이전트 실행, LLM 호출, 도구 호출을 추적 |
# | **Custom Span** | 개발자가 직접 정의하는 추적 단위 |
#
# ### 자동으로 추적되는 항목
# - `Runner.run()` 전체 실행 → **Trace**
# - 에이전트 실행 → **Agent Span**
# - LLM 호출 → **Generation Span**
# - 도구 호출 → **Function Span**
# - 가드레일 → **Guardrail Span**
# - 핸드오프 → **Handoff Span**

# %%
import os
from dotenv import load_dotenv

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
import openai

Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 자동 트레이싱 (기본 동작)
#
# 트레이싱은 **기본적으로 활성화**되어 있습니다.
# 별도 설정 없이도 모든 에이전트 실행이 자동으로 추적됩니다.

# %%
from agents import Agent, Runner, function_tool

@function_tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    try:
        result = eval(expression)
        return f"계산 결과: {expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"

math_agent = Agent(
    name="수학도우미",
    instructions="수학 계산을 도와주는 도우미입니다. calculate 도구를 사용하세요.",
    model=Model,
    tools=[calculate],
)

# 이 실행은 자동으로 트레이싱됩니다
result = await Runner.run(math_agent, "15 * 37 + 128을 계산해주세요.")

print(f"결과: {result.final_output}")

print("\n✅ 위 실행은 자동으로 OpenAI Dashboard에서 추적할 수 있습니다.")
print("https://platform.openai.com/traces 에서 확인하세요.")

# %% [markdown]
# ## 2. 커스텀 Trace로 여러 실행 묶기
#
# `trace()` 컨텍스트 매니저를 사용하면 여러 에이전트 실행을 **하나의 트레이스**로 묶을 수 있습니다.

# %%
from agents import trace

summarizer = Agent(
    name="요약기",
    instructions="주어진 텍스트를 한 문장으로 요약하세요.",
    model=Model,
)

critic = Agent(
    name="평가기",
    instructions="주어진 요약의 품질을 '좋음', '보통', '나쁨'으로 평가하고 이유를 설명하세요.",
    model=Model,
)

original_text = """
인공지능(AI)은 인간의 학습능력, 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 과학의 세부 분야입니다.
최근 대규모 언어 모델(LLM)의 발전으로 자연어 처리, 코드 생성, 창작 활동 등 다양한 분야에서
혁신적인 성과를 거두고 있습니다.
"""

print(original_text)

# 하나의 트레이스로 여러 실행을 묶기
with trace("요약 및 평가 워크플로우"):
    # Step 1: 요약
    summary_result = await Runner.run(summarizer, f"다음 텍스트를 요약해주세요:\n{original_text}")
    print(f"요약: {summary_result.final_output}")

    # Step 2: 요약 평가
    eval_result = await Runner.run(critic, f"다음 요약을 평가해주세요:\n{summary_result.final_output}")
    print(f"평가: {eval_result.final_output}")

print("\n두 실행이 '요약 및 평가 워크플로우'라는 하나의 트레이스로 묶였습니다.")

# %% [markdown]
# ## 3. Trace에 메타데이터 추가
#
# `trace()`에 `group_id`를 추가하면 관련 트레이스를 그룹으로 묶어 관리할 수 있습니다.

# %%
# group_id로 관련 트레이스를 그룹화
with trace("고객 요청 처리", group_id="customer-session-abc123"):
    result = await Runner.run(
        math_agent,
        "100 * 5 + 20000을 계산해주세요."
    )
    print(f"결과: {result.final_output}")

print("\n✅ group_id='customer-session-abc123'으로 이 트레이스가 태그되었습니다.")
print("   동일 세션의 여러 트레이스를 그룹으로 조회할 수 있습니다.")

# %% [markdown]
# ## 4. 트레이싱 비활성화
#
# 필요에 따라 트레이싱을 비활성화할 수 있습니다.

# %%
from agents import RunConfig

# 방법 1: 특정 실행에서만 비활성화
result = await Runner.run(
    math_agent,
    "1 + 1을 계산해주세요.",
    run_config=RunConfig(tracing_disabled=True),
)
print(f"결과: {result.final_output}")
print("✅ 이 실행은 트레이싱되지 않았습니다.")

# %%
# 방법 2: 환경 변수로 전역 비활성화 (코드 실행 전에 설정)
# import os
# os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# 방법 3: SDK 함수로 전역 비활성화
# from agents import set_tracing_disabled
# set_tracing_disabled(True)

print("\n📌 트레이싱 비활성화 방법:")
print("   1. RunConfig(tracing_disabled=True) - 특정 실행만")
print("   2. 환경변수 OPENAI_AGENTS_DISABLE_TRACING=1 - 전역")
print("   3. set_tracing_disabled(True) - 전역 (코드)")

# %% [markdown]
# ## 5. 민감 데이터 제외
#
# LLM 입출력 등 민감한 데이터를 트레이스에서 제외할 수 있습니다.

# %%
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper, function_tool
from agents import RunConfig

# 민감 데이터를 포함한 컨텍스트 정의
@dataclass
class PatientContext:
    patient_id: str        # 환자 고유 ID
    name: str              # 환자 이름
    ssn: str               # 주민등록번호
    diagnosis: str         # 진단명
    prescription: str      # 처방전

# 민감 데이터를 다루는 도구
@function_tool
async def get_patient_info(wrapper: RunContextWrapper[PatientContext]) -> str:
    """환자의 의료 정보를 조회합니다."""
    ctx = wrapper.context
    return (
        f"환자명: {ctx.name}\n"
        f"주민번호: {ctx.ssn}\n"
        f"진단명: {ctx.diagnosis}\n"
        f"처방전: {ctx.prescription}"
    )

# 에이전트 정의
medical_agent = Agent[PatientContext](
    name="의료 정보 에이전트",
    instructions="환자의 의료 정보를 조회하고 요약해주세요.",
    tools=[get_patient_info],
)

# 민감한 실제 환자 데이터
patient = PatientContext(
    patient_id="P-20240001",
    name="홍길동",
    ssn="800101-1234567",       # 주민등록번호
    diagnosis="고혈압 2기",       # 진단명
    prescription="암로디핀 5mg 1일 1회"  # 처방전
)

# 민감 데이터를 트레이스에서 제외
# → LLM에 전달되는 입출력(주민번호, 진단명 등)이 트레이스 로그에 기록되지 않음
result = await Runner.run(
    medical_agent,
    "현재 환자의 의료 정보를 조회하고 요약해주세요.",
    context=patient,
    run_config=RunConfig(trace_include_sensitive_data=False),
)

print(f"결과: {result.final_output}")
print("✅ 주민번호, 진단명 등 민감 의료 데이터가 트레이스 로그에 기록되지 않았습니다.")

# %% [markdown]
# ### 정리
#
# | 기능 | 코드 | 설명 |
# |------|------|------|
# | 자동 트레이싱 | (기본 활성화) | 모든 실행이 자동 추적됨 |
# | 커스텀 트레이스 | `with trace("이름"):` | 여러 실행을 하나의 트레이스로 묶기 |
# | 그룹화 | `trace("이름", group_id="id")` | 관련 트레이스를 그룹으로 관리 |
# | 비활성화 | `RunConfig(tracing_disabled=True)` | 특정 실행의 트레이싱 끄기 |
# | 민감 데이터 제외 | `RunConfig(trace_include_sensitive_data=False)` | LLM 입출력 데이터 제외 |
# | 대시보드 확인 | https://platform.openai.com/traces | OpenAI에서 트레이스 시각적 확인 |
#
# **활용 사례:**
# - 에이전트 동작 디버깅 (어떤 도구를 호출했는지, 어떤 핸드오프가 발생했는지)
# - 성능 모니터링 (각 단계별 소요 시간)
# - 비용 분석 (토큰 사용량 추적)
# - 품질 평가 (응답 품질 분석을 위한 데이터 수집)
