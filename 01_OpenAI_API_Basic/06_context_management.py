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
# # 06. Context Management (컨텍스트 관리)
#
# 에이전트 실행 중 **사용자 정보(user id, 권한 등), DB 연결 객체, API 키, 설정값** 등을 도구 함수에 전달해야 할 때가 있습니다.
# `RunContextWrapper`를 사용하면 이러한 데이터를 **의존성 주입(Dependency Injection)** 방식으로 전달할 수 있습니다.
#
# ### 핵심 개념
#
# | 개념 | 설명 |
# |------|------|
# | **Local Context** | 도구 함수에서 접근 가능한 로컬 데이터 (LLM에 전송되지 않음) |
# | **RunContextWrapper[T]** | 컨텍스트를 감싸는 래퍼 클래스 |
# | **ToolContext[T]** | RunContextWrapper + 도구 메타데이터 (도구 이름, 호출 ID 등) |
#
# ### 중요: 컨텍스트는 LLM에 전송되지 않습니다!
# 컨텍스트 객체는 **로컬에서만** 사용됩니다. 보안이 필요한 정보(API 키, DB 연결 등)를
# 안전하게 도구에 전달할 수 있습니다.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import openai

Model = "gpt-5-mini"

# %% [markdown]
# ## 1. 기본 컨텍스트 사용법
#
# `@dataclass`로 컨텍스트 객체를 정의하고, 도구 함수에서 `RunContextWrapper`를 통해 접근합니다.

# %%
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper, function_tool

# ============================================================
# 컨텍스트 객체 정의
# - 에이전트 실행 전체에서 공유될 사용자 정보를 담는 데이터 클래스
# - @dataclass 데코레이터로 __init__, __repr__ 등을 자동 생성
# ============================================================
@dataclass
class UserContext:
    user_id: str       # 사용자 고유 식별자
    user_name: str     # 사용자 이름
    is_premium: bool   # 프리미엄 멤버십 여부
    
# ============================================================
# 도구 함수 1: 사용자 프로필 조회
# - @function_tool: LLM이 호출할 수 있는 도구로 등록
# - RunContextWrapper[UserContext]: 타입 힌트로 컨텍스트 타입을 명시
#   → 실제 컨텍스트 객체는 wrapper.context로 접근
# ============================================================
@function_tool
async def get_user_profile(wrapper: RunContextWrapper[UserContext]) -> str:
    """현재 사용자의 프로필 정보를 조회합니다."""
    
    # RunContextWrapper에서 실제 UserContext 객체 추출
    ctx = wrapper.context
    
    # is_premium 값에 따라 멤버십 등급 문자열 결정
    membership = "프리미엄" if ctx.is_premium else "일반"
    
    # 사용자 정보를 포맷팅하여 반환 (LLM이 응답에 활용)
    return f"사용자: {ctx.user_name} (ID: {ctx.user_id}), 멤버십: {membership}"

# ============================================================
# 도구 함수 2: 개인화 추천 항목 반환
# - 컨텍스트의 is_premium 값에 따라 다른 추천 콘텐츠 제공
# - LLM에게 파라미터를 요구하지 않고 컨텍스트에서 직접 정보를 가져옴
#   → 민감한 사용자 정보가 LLM 프롬프트에 노출되지 않는 보안상 이점
# ============================================================
@function_tool
async def get_recommendations(wrapper: RunContextWrapper[UserContext]) -> str:
    """사용자에 맞는 추천 항목을 반환합니다."""
    
    # RunContextWrapper에서 실제 UserContext 객체 추출
    ctx = wrapper.context
    
    # 멤버십 등급에 따라 차별화된 추천 콘텐츠 반환
    if ctx.is_premium:
        # 프리미엄 사용자: 고급 콘텐츠 및 1:1 서비스 추천
        return f"{ctx.user_name}님을 위한 프리미엄 추천: AI 고급 과정, 1:1 멘토링, 전용 워크숍"
    else:
        # 일반 사용자: 입문 콘텐츠 및 커뮤니티 기반 서비스 추천
        return f"{ctx.user_name}님을 위한 추천: AI 입문 과정, 온라인 세미나, 커뮤니티 스터디"

# %% [markdown]
# ## 2. 에이전트 생성 및 실행
#
# `Agent[T]`로 컨텍스트 타입을 지정하고, `Runner.run()`의 `context` 파라미터로 전달합니다.

# %%
# 컨텍스트 타입이 지정된 에이전트
agent = Agent[UserContext](
    name="개인화 도우미",
    instructions="""당신은 사용자 맞춤형 도우미입니다.
사용자의 프로필과 추천 정보를 활용하여 답변하세요.
도구를 사용하여 사용자 정보를 확인할 수 있습니다.""",
    model=Model,
    tools=[get_user_profile, get_recommendations],
)

# 프리미엄 사용자 컨텍스트
premium_user = UserContext(
    user_id="USR-001",
    user_name="김영희",
    is_premium=True,
)

# 컨텍스트를 전달하며 에이전트 실행
result = await Runner.run(
    starting_agent=agent,
    input="내 프로필 정보와 추천 과정을 알려주세요.",
    context=premium_user,  # 컨텍스트 전달
)
print(f"결과: {result.final_output}")

# %% [markdown]
# ## 3. 동일한 에이전트, 다른 컨텍스트
#
# 같은 에이전트를 **다른 사용자 컨텍스트**로 실행하면 개인화된 결과를 얻습니다.

# %%
# 일반 사용자 컨텍스트
regular_user = UserContext(
    user_id="USR-002",
    user_name="박철수",
    is_premium=False,
)

result = await Runner.run(
    starting_agent=agent,
    input="나한테 맞는 추천 과정을 알려주세요.",
    context=regular_user,  # 다른 컨텍스트 전달
)
print(f"결과: {result.final_output}")

# %% [markdown]
# ## 4. 실전 예제: 컨텍스트에 서비스 의존성 주입
#
# 실무에서는 DB 연결, API 클라이언트 등의 **서비스 의존성**을 컨텍스트에 담아 전달합니다.

# %%
@dataclass
class AppContext:
    user_id: str
    user_name: str
    order_db: dict  # 간단한 예시로 딕셔너리 사용 (실제로는 DB 커넥션, API 클라이언트 등이 들어갑니다)
    settings: dict

# 주문 조회 도구 - 컨텍스트의 DB를 사용
@function_tool
async def lookup_order(wrapper: RunContextWrapper[AppContext], order_id: str) -> str:
    """주문 ID로 주문 정보를 조회합니다."""
    ctx = wrapper.context
    order = ctx.order_db.get(order_id)
    if order:
        return f"{ctx.user_name}님의 주문 [{order_id}]: {order['item']} - {order['status']}"
    return f"주문 {order_id}를 찾을 수 없습니다."

@function_tool
async def get_shipping_info(wrapper: RunContextWrapper[AppContext], order_id: str) -> str:
    """주문의 배송 정보를 조회합니다."""
    ctx = wrapper.context
    order = ctx.order_db.get(order_id)
    if order and "tracking" in order:
        return f"배송 추적번호: {order['tracking']}, 예상 도착: {order.get('eta', '미정')}"
    return "배송 정보가 아직 없습니다."

# 에이전트 생성
order_agent = Agent[AppContext](
    name="주문 관리 도우미",
    instructions="""당신은 주문 관리 도우미입니다.
사용자의 주문을 조회하고 배송 정보를 안내합니다.""",
    model=Model,
    tools=[lookup_order, get_shipping_info],
)

# 애플리케이션 컨텍스트 생성 (DB, 설정 등 포함)
app_context = AppContext(
    user_id="USR-001",
    user_name="김영희",
    order_db={
        "ORD-100": {"item": "Python 교재", "status": "배송 중", "tracking": "KR1234567890", "eta": "2025-03-01"},
        "ORD-101": {"item": "AI 키보드", "status": "준비 중"},
    },
    settings={"max_orders": 10, "currency": "KRW"},
)

result = await Runner.run(
    starting_agent=order_agent,
    input="ORD-100 주문 상태랑 배송 정보 알려주세요.",
    context=app_context,
)
print(f"결과: {result.final_output}")

# %% [markdown]
# ## 5. ToolContext로 도구 메타데이터 접근
#
# `ToolContext`는 `RunContextWrapper`를 확장하여 **도구 자체의 메타데이터**에도 접근할 수 있습니다.
# - `ctx.tool_name`: 호출된 도구 이름
# - `ctx.tool_call_id`: 도구 호출 고유 ID
# - `ctx.tool_arguments`: 원본 인자 문자열

# %%
from agents.tool_context import ToolContext

# ============================================================
# ToolContext = RunContextWrapper를 상속한 클래스
# RunContextWrapper 기본 기능 + 도구 실행 메타데이터 추가 제공
# ============================================================
@function_tool
async def debug_tool(ctx: ToolContext[UserContext], query: str) -> str:
    """디버깅용 도구 - 도구 메타데이터를 출력합니다."""
    
    # ToolContext 전용 메타데이터 출력 (RunContextWrapper에는 없는 정보)
    print(f"  [디버그] 도구 이름: {ctx.tool_name}")       # 예: "debug_tool"
    print(f"  [디버그] 호출 ID: {ctx.tool_call_id}")      # 예: "call_abc123"
    print(f"  [디버그] 원본 인자: {ctx.tool_arguments}")  # 예: '{"query": "파이썬..."}'
    
    # RunContextWrapper와 동일하게 ctx.context로 UserContext 접근
    print(f"  [디버그] 사용자: {ctx.context.user_name}")
    
    return f"'{query}'에 대한 검색 결과입니다."

# ============================================================
# Agent[UserContext]: 이 에이전트가 UserContext 타입을 사용함을 명시
# → 등록된 도구들도 동일한 UserContext 타입을 사용해야 함
# ============================================================
debug_agent = Agent[UserContext](
    name="디버그 에이전트",
    instructions="사용자의 질문에 debug_tool을 사용하여 답변하세요.",
    model=Model,
    tools=[debug_tool],  # ToolContext[UserContext]를 사용하는 도구 등록
)

# ============================================================
# Runner.run(): 에이전트 실행
#     → 내부적으로 RunContextWrapper/ToolContext에 자동으로 래핑되어 전달됨
# ============================================================
result = await Runner.run(
    starting_agent=debug_agent,
    input="파이썬 비동기 프로그래밍에 대해 알려주세요.",
    context=premium_user,  # UserContext 인스턴스 → ctx.context로 접근 가능
)

# result.final_output: 에이전트가 최종적으로 반환한 텍스트 응답
print(f"\n결과: {result.final_output}")

# %% [markdown]
# ### 정리
#
# | 개념 | 코드 | 설명 |
# |------|------|------|
# | 컨텍스트 정의 | `@dataclass class MyContext` | 전달할 데이터/의존성 정의 |
# | 에이전트 타입 | `Agent[MyContext]` | 컨텍스트 타입 지정 |
# | 도구에서 접근 | `wrapper: RunContextWrapper[MyContext]` | 도구 함수의 첫 번째 인자 |
# | 실행 시 전달 | `Runner.run(..., context=my_ctx)` | context 파라미터로 전달 |
# | 도구 메타데이터 | `ctx: ToolContext[MyContext]` | 도구 이름, 호출 ID 등 추가 접근 |
#
# **활용 사례:**
# - 사용자별 개인화 (프로필, 권한, 설정)
# - DB 연결/API 클라이언트 주입
# - 로깅/감사 추적 (audit trail)
# - 요청별 설정값 전달
