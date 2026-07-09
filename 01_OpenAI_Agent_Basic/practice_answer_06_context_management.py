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
# # 실습 문제 모범답안 - 06. Context Management
#
# `06_context_management` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# **쇼핑몰 개인화 도우미 (RunContextWrapper 응용)**:
# 본문의 의존성 주입 4단계를 그대로 따라 쇼핑몰 도우미를 만드세요.
#
# - ① `ShopContext(user_name, is_vip, cart)` 정의
# - ④ 장바구니 요약 도구 `summarize_cart(wrapper)`: `cart` 목록과 개수 반환,
#   `is_vip`이면 `"VIP 10% 할인 적용 예정"` 안내
# - ② 같은 에이전트를 VIP / 일반 두 컨텍스트로 실행해 개인화 확인
#   (③ 래핑·주입은 SDK가 자동 처리)

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## ① Context 정의 → ④ 도구에서 접근
#
# - 장바구니와 VIP 여부는 **LLM에게 묻지 않고** 컨텍스트에서 직접 꺼냅니다.
# - 따라서 사용자 정보가 프롬프트에 노출되지 않습니다. (Local Context)

# %%
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper, function_tool

# ① Context 정의
@dataclass
class ShopContext:
    user_name: str
    is_vip: bool
    cart: list


# ④ 도구에서 접근 - wrapper.context로 장바구니 정보를 직접 꺼냄
@function_tool
async def summarize_cart(wrapper: RunContextWrapper[ShopContext]) -> str:
    """현재 사용자의 장바구니를 요약한다."""
    ctx = wrapper.context
    summary = f"{ctx.user_name}님의 장바구니 ({len(ctx.cart)}개): {', '.join(ctx.cart)}"
    if ctx.is_vip:
        summary += " / VIP 10% 할인 적용 예정"
    return summary

# %% [markdown]
# ## ② 타입 지정·전달 → 개인화 확인
#
# 같은 에이전트, 같은 질문에 **컨텍스트만 바꿔서** 실행합니다.
# (③ 래핑·주입은 SDK가 자동 처리하므로 코드가 없습니다)

# %%
# ② 타입 지정
shop_agent = Agent[ShopContext](
    name="쇼핑 도우미",
    instructions="쇼핑몰 도우미입니다. 도구로 장바구니를 확인해서 간결하게 답변하세요.",
    model=Model,
    tools=[summarize_cart],
)

# ② 전달 - VIP 고객
vip_user = ShopContext(user_name="김영희", is_vip=True, cart=["노트북", "무선 마우스", "USB 허브"])
result = await Runner.run(shop_agent, "제 장바구니 요약해 주세요.", context=vip_user)
print("VIP :", result.final_output)

# ② 전달 - 일반 고객 (같은 에이전트, 같은 질문, 다른 컨텍스트)
normal_user = ShopContext(user_name="박철수", is_vip=False, cart=["키보드", "모니터 받침대"])
result = await Runner.run(shop_agent, "제 장바구니 요약해 주세요.", context=normal_user)
print("일반:", result.final_output)

# %% [markdown]
# ### 정리
#
# | 단계 | 이 문제에서의 코드 |
# |------|------|
# | ① Context 정의 | `@dataclass class ShopContext: user_name / is_vip / cart` |
# | ② 타입 지정·전달 | `Agent[ShopContext](...)`, `Runner.run(..., context=vip_user)` |
# | ③ 래핑·주입 | (SDK 자동 - 코드 없음) |
# | ④ 도구에서 접근 | `summarize_cart(wrapper)` → `wrapper.context.cart`, `.is_vip` |
#
# - VIP는 할인 안내가 포함되고 일반은 포함되지 않음 → **같은 에이전트 + 다른 컨텍스트 = 개인화**
# - 장바구니·VIP 여부는 도구가 직접 꺼내므로 **프롬프트에 노출되지 않고 토큰도 소모하지 않습니다.**
