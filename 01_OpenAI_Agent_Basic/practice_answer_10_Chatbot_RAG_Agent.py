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
# # 실습 문제 모범답안 - 10. Chatbot RAG Agent
#
# `10_Chatbot_RAG_Agent` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# **고객 지원 챗봇 (스트리밍 + 함수 도구 + RAG 종합)**
#
# 1. **주문 조회 도구**: `get_order_status(order_id)` - 더미 주문 데이터에서 상태 조회
# 2. **RAG 도구**: 본문의 Vector Store(`vs_id`)를 `FileSearchTool`로 연결해 FAQ 답변
# 3. **스트리밍 대화 루프**: `Runner.run_streamed()` + `ResponseTextDeltaEvent`,
#    `messages`에 대화 기록 누적, `"exit"` 입력 시 종료
#

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 주문 조회 도구 + Vector Store 연결
#
# - 주문 상태는 외부 API 대신 **더미 dict**로 대체합니다. (도구 호출 여부 확인이 목적)
# - Vector Store는 본문에서 만든 `notebook-faq-kb`를 이름으로 찾아 재사용합니다.
#

# %%
import openai
from agents import Agent, Runner, function_tool, FileSearchTool
from openai.types.responses import ResponseTextDeltaEvent

# 더미 주문 데이터
ORDERS = {
    "ORD-1001": "배송 중 (내일 도착 예정)",
    "ORD-1002": "상품 준비 중",
    "ORD-1003": "배송 완료 (어제 수령)",
}

@function_tool
def get_order_status(order_id: str) -> str:
    """주문 번호로 주문 상태를 조회한다."""
    print(f"\n[도구 실행] get_order_status - 주문번호: {order_id}")
    return ORDERS.get(order_id, f"주문 {order_id}을(를) 찾을 수 없습니다.")


# 본문에서 만든 Vector Store를 이름으로 찾아 재사용
client = openai.OpenAI()
VS_NAME = "notebook-faq-kb"

existing = [vs for vs in client.vector_stores.list() if vs.name == VS_NAME]
if not existing:
    raise RuntimeError("Vector Store가 없습니다. 본문의 'Vector Store 초기 설정' 셀을 먼저 실행하세요.")

vs_id = existing[0].id
print(f"Vector Store 재사용: {vs_id}")

# %% [markdown]
# ## 2. 에이전트 정의 + 스트리밍 대화 루프
#
# - 본문의 대화 루프와 같은 구조입니다. 도구 구성만 `get_order_status` + `FileSearchTool`로 바뀌었습니다.
# - `messages`에 사용자/에이전트 발화를 누적하므로 "그 주문 언제 도착해요?" 같은 **멀티턴 질문**도 처리됩니다.
#

# %%
messages = []  # 대화 기록 누적 리스트

support_agent = Agent(
    model=Model,
    name="고객 지원 에이전트",
    instructions=(
        "당신은 친절한 고객 지원 에이전트입니다. "
        "주문 상태 질문은 get_order_status 도구로 조회하세요. "
        "환불·계정·비밀번호·보안 등 FAQ 질문은 file_search 도구로 "
        "지식 베이스를 검색한 뒤 답하세요. 답은 간결하게 하세요."
    ),
    tools=[
        get_order_status,                          # 주문 상태 조회 도구
        FileSearchTool(vector_store_ids=[vs_id]),  # FAQ 지식 베이스 검색 도구
    ],
)

while True:
    user_input = input("\n사용자: ")
    if user_input == "exit":
        print("Bye")
        break

    # 사용자 메시지를 대화 기록에 추가
    messages.append({"role": "user", "content": user_input})

    print("\n에이전트: ", end="", flush=True)

    # 스트리밍 방식으로 에이전트 실행 (전체 대화 기록 전달)
    result = Runner.run_streamed(support_agent, input=messages)
    full_response = ""

    # 텍스트 델타 이벤트만 골라 토큰 단위로 실시간 출력
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            print(delta, end="", flush=True)
            full_response += delta

    # 완성된 응답을 대화 기록에 추가
    messages.append({"role": "assistant", "content": full_response})

# %% [markdown]
# ### 정리
#
# | 구성 요소 | 코드 | 역할 |
# |-----------|------|------|
# | 함수 도구 | `@function_tool get_order_status` | 더미 주문 데이터에서 상태 조회 |
# | RAG 도구 | `FileSearchTool(vector_store_ids=[vs_id])` | FAQ 지식 베이스 검색 |
# | 스트리밍 | `Runner.run_streamed()` + `ResponseTextDeltaEvent` | 토큰 단위 실시간 출력 |
# | 멀티턴 | `messages` 리스트에 user/assistant 발화 누적 | 이전 대화 기억 |
#
# **실행 흐름 예시:**
# - "환불 정책이 어떻게 되나요?" 👉 file_search → 30일 이내 환불 안내
# - "주문번호 ORD-1001 상태 알려줘" 👉 get_order_status 실행 → "배송 중"
# - "그 주문 언제쯤 도착할까요?" 👉 대화 기록에서 ORD-1001을 기억하고 답변
#
