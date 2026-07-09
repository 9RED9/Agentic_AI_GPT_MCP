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
# 도구 함수에 **사용자 정보(user_id·권한), DB 연결, API 키, 설정값** 등
# 실행마다 달라지는 데이터를 효율적으로 전달하는 방법입니다.
# `RunContextWrapper`를 사용한 **의존성 주입(Dependency Injection)** 방식입니다.
#
# ### 의존성 주입 흐름 (4단계)
#
# | 단계 | 하는 일 | 코드 | 누가 |
# |---|---|---|---|
# | ① Context 정의 | 전달할 데이터 클래스 작성 | `@dataclass class UserContext: ...` | 사용자 작성 |
# | ② 타입 지정·전달 | 에이전트에 타입 지정, 실행 시 전달 | `Agent[UserContext](...)`, `Runner.run(..., context=premium_user)` | 사용자 작성 |
# | ③ 래핑·주입 | `RunContextWrapper`로 감싸 각 도구 호출에 전달 | (자동) | **SDK가 자동 처리** |
# | ④ 도구에서 접근 | 도구 첫 인자로 받아 꺼내 사용 | `def tool(wrapper: RunContextWrapper[...])` → `wrapper.context.user_name` | 사용자 작성 |
#
# ### 중요: 컨텍스트는 LLM에 전송되지 않습니다! (Local Context)
#
# - 도구가 필요한 정보를 컨텍스트에서 **직접** 꺼내므로,
#   API 키·DB 연결·개인정보가 **프롬프트에 노출되지 않고 토큰도 소모하지 않습니다.**
#
# ### 학습 내용
#
# 1. 의존성 주입 4단계 — Context 정의부터 도구에서 접근까지
# 2. 같은 에이전트 + 다른 컨텍스트 = **사용자별 개인화**
# 3. 컨텍스트가 LLM에 전송되지 않는 것을 **코드로 확인**
# 4. 실전: DB 등 **서비스 의존성 주입**
# 5. `ToolContext` — 도구 메타데이터(도구 이름, 호출 ID, 원본 인자) 접근

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. ① Context 정의 → ④ 도구에서 접근
#
# - **① Context 정의**: `@dataclass`로 전달할 데이터를 정의합니다.
# - **④ 도구에서 접근**: 도구 함수의 **첫 번째 인자**로 `RunContextWrapper[UserContext]`를 받고,
#   `wrapper.context`로 실제 객체를 꺼내 사용합니다.
#
# 도구가 LLM에게 사용자 정보를 물어보지 않고 **컨텍스트에서 직접** 가져오는 것에 주목하세요.

# %%
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper, function_tool

# ① Context 정의 - 실행마다 달라지는 데이터를 담는 클래스
@dataclass
class UserContext:
    user_id: str
    user_name: str
    is_premium: bool


# ④ 도구에서 접근 - 첫 번째 인자로 wrapper를 받고, wrapper.context로 꺼내 사용
@function_tool
async def get_recommendations(wrapper: RunContextWrapper[UserContext]) -> str:
    """현재 사용자에게 맞는 추천 과정을 반환한다."""
    ctx = wrapper.context   # 실제 UserContext 객체
    if ctx.is_premium:
        return f"{ctx.user_name}님(프리미엄) 추천: AI 고급 과정, 1:1 멘토링, 전용 워크숍"
    return f"{ctx.user_name}님(일반) 추천: AI 입문 과정, 온라인 세미나, 커뮤니티 스터디"

# %% [markdown]
# ## 2. ② 타입 지정·전달 (③ 래핑·주입은 SDK가 자동 처리)
#
# - `Agent[UserContext](...)`: 이 에이전트가 사용할 컨텍스트 타입 지정
# - `Runner.run(..., context=premium_user)`: 실행 시 컨텍스트 객체 전달
# - **③ 래핑·주입**: SDK가 `RunContextWrapper`로 감싸 각 도구 호출에 자동 전달 — 우리가 할 일 없음

# %%
# ② 타입 지정 - Agent[UserContext]
agent = Agent[UserContext](
    name="개인화 도우미",
    instructions="사용자 맞춤 도우미입니다. 도구로 추천 정보를 확인해서 간결하게 답변하세요.",
    model=Model,
    tools=[get_recommendations],
)

# ② 전달 - context 파라미터로 전달 (③ 래핑·주입은 SDK가 자동 처리)
premium_user = UserContext(user_id="USR-001", user_name="김영희", is_premium=True)

result = await Runner.run(
    agent,
    "저에게 맞는 과정을 추천해 주세요.",
    context=premium_user,
)

print(result.final_output)

# %% [markdown]
# ## 3. 같은 에이전트 + 다른 컨텍스트 = 사용자별 개인화
#
# 에이전트와 질문은 완전히 동일하고 **컨텍스트만 다릅니다.**
# 도구가 컨텍스트에서 다른 정보를 꺼내므로 답변이 개인화됩니다.

# %%
# 일반 사용자 컨텍스트로 같은 에이전트 실행
regular_user = UserContext(user_id="USR-002", user_name="박철수", is_premium=False)

result = await Runner.run(
    agent,
    "저에게 맞는 과정을 추천해 주세요.",   # 같은 질문
    context=regular_user,              # 다른 컨텍스트
)

print(result.final_output)

# %% [markdown]
# ## 4. 확인: 컨텍스트는 LLM에 전송되지 않는다 (Local Context)
#
# 컨텍스트를 전달해도 **LLM 프롬프트에 자동으로 포함되는 것이 아닙니다.**
# 도구가 없는 에이전트에 같은 컨텍스트를 넘기고 이름을 물어보면 → **모른다고 답합니다.**
#
# - LLM이 아는 것은 오직 **도구가 컨텍스트에서 꺼내 반환해 준 문자열**뿐입니다.
# - 그래서 API 키·DB 연결·개인정보를 담아도 프롬프트에 노출되지 않고, 토큰도 소모하지 않습니다.

# %%
# 도구가 없는 에이전트 - 컨텍스트를 전달해도 LLM은 그 내용을 볼 수 없음
no_tool_agent = Agent[UserContext](
    name="도구없는_에이전트",
    instructions="간결하게 답변하세요.",
    model=Model,
)

result = await Runner.run(no_tool_agent, "제 이름이 뭔가요?", context=premium_user)
print(result.final_output)  # 이름(김영희)을 모른다고 답함 → 컨텍스트는 프롬프트에 포함되지 않음


# %% [markdown]
# ## 5. 실전: 서비스 의존성 주입
#
# 실무에서는 **DB 연결, API 클라이언트** 같은 서비스 객체를 컨텍스트에 담아 전달합니다.
# (여기서는 DB 대신 딕셔너리로 시연 — 구조는 동일)

# %%
@dataclass
class AppContext:
    user_name: str
    order_db: dict   # 실제로는 DB 커넥션, API 클라이언트 등이 들어감

# 도구는 컨텍스트에 주입된 DB를 사용 (LLM에게 DB 정보를 요구하지 않음)
@function_tool
async def lookup_order(wrapper: RunContextWrapper[AppContext], order_id: str) -> str:
    """주문 ID로 주문 정보를 조회한다."""
    ctx = wrapper.context
    order = ctx.order_db.get(order_id)
    if order:
        return f"{ctx.user_name}님의 주문 [{order_id}]: {order['item']} - {order['status']}"
    return f"주문 {order_id}를 찾을 수 없습니다."


order_agent = Agent[AppContext](
    name="주문 도우미",
    instructions="주문 조회 도우미입니다. 간결하게 답변하세요.",
    model=Model,
    tools=[lookup_order],
)

# 애플리케이션 컨텍스트 (DB 등 서비스 의존성 포함)
app_context = AppContext(
    user_name="김영희",
    order_db={
        "ORD-100": {"item": "Python 교재", "status": "배송 중"},
        "ORD-101": {"item": "AI 키보드", "status": "준비 중"},
    },
)

result = await Runner.run(order_agent, "ORD-100 주문 상태 알려주세요.", context=app_context)
print(result.final_output)

# %% [markdown]
# ## 6. ToolContext — 도구 메타데이터 접근
#
# **"지금 어떤 도구가, 어떤 인자로 호출됐는가"** 를 도구 안에서 알아야 할 때 사용합니다.
# (로깅, 감사 추적(audit trail), 디버깅 등)
#
# **상속 관계**: `ToolContext`는 `RunContextWrapper`를 상속 + 도구 메타데이터 3가지 추가
#
# | 추가 정보 | 설명 |
# |---|---|
# | `ctx.tool_name` | 호출된 도구 이름 |
# | `ctx.tool_call_id` | 이번 호출의 고유 ID |
# | `ctx.tool_arguments` | LLM이 넘긴 원본 인자(JSON) |
#
# 기존 `ctx.context.user_name` 접근은 그대로 사용할 수 있고, **도구 시그니처만 교체**하면 됩니다.

# %%
from agents.tool_context import ToolContext

# 시그니처만 RunContextWrapper → ToolContext로 교체 (에이전트/실행 코드는 동일)
@function_tool
async def debug_tool(ctx: ToolContext[UserContext], query: str) -> str:
    """디버깅용 검색 도구"""
    print("[디버그] 도구 이름:", ctx.tool_name)        # 호출된 도구 이름
    print("[디버그] 호출 ID:", ctx.tool_call_id)       # 이번 호출의 고유 ID
    print("[디버그] 원본 인자:", ctx.tool_arguments)   # LLM이 넘긴 원본 인자(JSON)
    print("[디버그] 사용자:", ctx.context.user_name)   # 기존 접근은 그대로
    return f"'{query}' 검색 결과: 관련 자료 3건을 찾았습니다."


debug_agent = Agent[UserContext](
    name="디버그 에이전트",
    instructions="질문에 debug_tool을 사용해서 한두 문장으로 답변하세요.",
    model=Model,
    tools=[debug_tool],
)

result = await Runner.run(debug_agent, "파이썬 비동기 프로그래밍 자료를 찾아주세요.", context=premium_user)
print("\n결과:", result.final_output)

# %%

# %% [markdown]
# ### 실습 문제
#
# **쇼핑몰 개인화 도우미 (RunContextWrapper 응용)**:
# 본문의 의존성 주입 4단계를 그대로 따라 쇼핑몰 도우미를 만드세요.
#
# - ① `ShopContext(user_name: str, is_vip: bool, cart: list)` 정의
# - ④ 장바구니 요약 도구 `summarize_cart(wrapper)`: 컨텍스트의 `cart` 목록과 개수를 반환하되,
#   `is_vip`이면 `"VIP 10% 할인 적용 예정"`을 함께 안내
# - ② 같은 에이전트를 **VIP / 일반** 두 컨텍스트로 실행해 답변이 개인화되는지 확인하세요.
#   (③ 래핑·주입은 SDK가 자동 처리)
#
#
# ### 테스트 입력 예시
#
# * `"제 장바구니 요약해 주세요."`
#   * VIP 👉 장바구니 목록 + `"VIP 10% 할인 적용 예정"` 포함
#   * 일반 👉 장바구니 목록만 (할인 안내 없음)

# %%
