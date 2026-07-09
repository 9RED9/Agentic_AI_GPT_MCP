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
# # 10. Chatbot RAG Agent 구현

# %% [markdown]
# **OpenAI Agents SDK**로 채팅 에이전트를 단계별로 확장합니다.
# 도구 없는 단순 챗봇에서 시작해 날씨/웹검색/RAG 도구를 하나씩 붙여가며 완성합니다.
#
# ```
# 1. 스트리밍 대화     Runner.run_streamed() + 토큰 단위 출력
#       ↓
# 2. 도구 추가         + get_weather + WebSearchTool
#       ↓
# 3. RAG 추가          + FileSearchTool (Vector Store)
#       ↓
# 4. 웹 UI            → Flask app.py
# ```
#
# | 단계 | 내용 | 핵심 API / 도구 |
# |------|------|-----------------|
# | 1. Streaming 방식 호출 | 도구 없이 Agent + Runner로 대화 구현 (대화 기록 유지 + 토큰 단위 실시간 출력) | `Runner.run_streamed()`, `ResponseTextDeltaEvent` |
# | 2. 도구를 추가한 Chatbot | 날씨 조회 + 웹 검색 도구를 에이전트에 연결 | `@function_tool` get_weather, `WebSearchTool()` |
# | 3. RAG 도구 추가 | OpenAI Vector Store에 FAQ/KB 업로드 후 검색 도구 연결 | `FileSearchTool(vector_store_ids=[...])` |
# | 4. 웹 UI 연결 | 동일 에이전트를 Flask 웹 앱으로 구현한 예제 안내 | `11_rag-chat-flask/app.py` |

# %%
import openai

from dotenv import load_dotenv
load_dotenv() 

Model = "gpt-5.4-mini"

# %% [markdown]
# ### Streaming 방식 호출
# 도구 없이 Agent + Runner로 대화를 구현합니다.
# `messages` 리스트에 사용자/에이전트 발화를 누적해 대화 기록을 유지하고, `Runner.run_streamed()`로 토큰 단위 실시간 출력을 처리합니다.
# (비동기 호출 `Runner.run()`은 02_agent_runner 노트북 참고)

# %%
#  모델의 답변이 완성될 때까지 기다리지 않고, 한 토큰씩 실시간으로 출력되는 "스트리밍" 방식
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

messages = []  # 지금까지의 대화 내용을 저장할 리스트

agent = Agent(
    model=Model,
    name="여행 에이전트",
    instructions="당신은 훌륭한 여행 에이전트입니다. 사용자와 대화하면서 여행 계획을 도와주세요.",
)

while True:
    user_input = input("\n사용자: ")
    
    if user_input == "exit":
        print("Bye")
        break

    # 현재 사용자 발화를 messages 리스트에 추가
    messages.append({"role": "user", "content": user_input})

    # 에이전트의 답변 출력 시작
    print("\n여행 에이전트: ", end="", flush=True)

    # run_streamed: 모델이 응답을 실시간으로 보낼 수 있도록 함
    # 결과(result)는 이벤트 스트림(event stream)을 포함하며,
    # 이를 async for 루프를 통해 한 토큰씩 처리합니다.
    result = Runner.run_streamed(agent, input=messages)
    full_response = ""  # 전체 응답을 누적할 변수

    # ResponseTextDeltaEvent: 모델이 생성 중인 텍스트 일부를 전달하는 이벤트 타입
    # event.type == "raw_response_event" 일 때,
    # event.data.delta 에는 모델이 방금 생성한 텍스트 조각이 들어 있음
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""   # 새로 들어온 텍스트 조각
            print(delta, end="", flush=True) # 화면에 즉시 출력
            full_response += delta           # 전체 답변 문자열로 누적

    # 모델의 전체 응답을 대화 기록에 추가
    messages.append({"role": "assistant", "content": full_response})

# %% [markdown]
# ### 도구를 추가한 Chatbot Agent
#
# WebSearchTool - 모델이 응답을 생성하기 전에 웹에서 최신 정보를 검색할 수 있도록 허용합니다.  
# https://developers.openai.com/api/docs/guides/tools-web-search

# %%
import requests
from agents import Agent, Runner, function_tool, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent

@function_tool
def get_weather(위도: float, 경도: float) -> str:
    print(위도, 경도)
    print(f"Weather 함수 실행 - 도시: {위도, 경도}")
    latitude = 위도
    longitude = 경도
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m")
    data = response.json()
    return data['current']['temperature_2m']

messages = []

agent = Agent(
    model=Model,
    name="여행 에이전트",
    instructions="당신은 훌륭한 여행 에이전트입니다. 사용자와 대화하면서 여행 계획을 도와주세요.",
    tools=[get_weather, WebSearchTool()]
)

while True:
    user_input = input("\n사용자: ")

    if user_input == "exit":
        print("Bye")
        break

    messages.append({"role": "user", "content": user_input})

    print("\n여행 에이전트: ", end="", flush=True)

    result = Runner.run_streamed(agent, input=messages)
    full_response = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            print(delta, end="", flush=True)
            full_response += delta

    messages.append({"role": "assistant", "content": full_response})

# %% [markdown]
# ### RAG(검색 증강 생성) 도구 추가
#
# **RAG**는 지식 베이스(FAQ, 문서)를 검색해 사용자 질문에 관련된 내용을 맥락으로 제공하는 방식입니다.
# 여기서는 **OpenAI 관리형 Vector Store**에 FAQ와 지식 베이스 파일을 업로드하고,
# 에이전트의 **`FileSearchTool`** 로 연결합니다. 별도 임베딩 코드 없이 OpenAI가 파싱/청킹/임베딩/검색을 모두 처리합니다.
#
# - **셀 1 (초기 설정)**: Vector Store를 생성하고 `faqs.json`, `knowledgeBase.json`을 업로드 (최초 1회)
# - **셀 2 (에이전트)**: `FileSearchTool(vector_store_ids=[vs_id])`를 도구로 추가해 에이전트 실행

# %%
# ── Vector Store 초기 설정 (최초 1회 실행) ──
# faqs.json / knowledgeBase.json 을 OpenAI Vector Store에 업로드합니다.
# 이미 같은 이름의 Vector Store가 있으면 재사용하고 재업로드하지 않습니다.

import openai
from pathlib import Path

client = openai.OpenAI()

VS_NAME = "notebook-faq-kb"
DATA_DIR = Path.cwd() / "data"

# 같은 이름의 Vector Store가 이미 있으면 재사용
existing = [vs for vs in client.vector_stores.list() if vs.name == VS_NAME]

if existing:
    vs_id = existing[0].id
    print(f"기존 Vector Store 재사용: {vs_id}")
else:
    vs = client.vector_stores.create(name=VS_NAME)
    vs_id = vs.id
    print(f"Vector Store 생성: {vs_id}")

    files_to_upload = [
        open(DATA_DIR / "faqs.json", "rb"),
        open(DATA_DIR / "knowledgeBase.json", "rb"),
    ]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id,
            files=files_to_upload,
        )
        print(f"파일 업로드 완료 - 상태: {batch.status}  파일 수: {batch.file_counts.completed}")
    finally:
        for f in files_to_upload:
            f.close()

print(f"\nvs_id = '{vs_id}'")

# %% [markdown]
# ```
# faqs.json
# [
#   {"question": "비밀번호를 어떻게 재설정하나요?", "answer": "비밀번호를 재설정하려면 계정 설정 페이지에서 '비밀번호 찾기'를 클릭하세요."},
#   {"question": "환불 정책이 어떻게 되나요?", "answer": "구매일로부터 30일 이내 환불이 가능합니다. 주문 번호와 함께 고객 지원팀에 문의하세요."},
#   {"question": "전화 상담을 제공하나요?", "answer": "네. 월요일부터 금요일 오전 10시부터 오후 6시(IST)까지 전화 상담이 가능합니다."},
#   {"question": "이메일 주소는 어떻게 변경하나요?", "answer": "프로필 설정의 이메일 환경설정에서 이메일을 변경할 수 있습니다."},
#   {"question": "제 데이터는 안전한가요?", "answer": "네. 모든 사용자 데이터는 업계 표준 보안 방식으로 암호화되어 보호됩니다."}
# ]
# ```
#
# ```
# kowledgeBase.json
# [
#   {"question": "환불 정책이 어떻게 되나요?", "answer": "구매일로부터 30일 이내 모든 상품을 전액 환불받을 수 있습니다."},
#   {"question": "주문은 어디서 조회하나요?", "answer": "웹사이트의 주문 조회 페이지에서 주문 번호로 조회할 수 있습니다."}
# ]
#
# ```

# %%
from agents import Agent, Runner, FileSearchTool, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent

# 대화 히스토리 저장 리스트 (멀티턴 대화를 위해 메시지를 누적)
messages = []

# 여행 및 FAQ 처리 에이전트 정의
agent = Agent(
    model="gpt-5.4-mini",
    name="여행·지식 베이스 에이전트",
    instructions=(
        "당신은 여행 계획과 회사 FAQ를 도와주는 에이전트입니다. "
        "날씨가 필요하면 get_weather, 최신 정보는 WebSearchTool을 사용하세요. "
        "계정·환불·정책·비밀번호·보안 등 FAQ 관련 질문은 file_search 도구로 "
        "지식 베이스를 검색한 뒤 답하세요. 답은 간결하게 하세요."
    ),
    tools=[
        get_weather,                               # 날씨 조회 도구 (이전 셀에서 정의)
        WebSearchTool(),                           # 최신 정보 웹 검색 도구
        FileSearchTool(vector_store_ids=[vs_id]),  # 회사 FAQ 지식 베이스 검색 도구
    ],
)

# 사용자가 'exit' 입력 시 종료되는 대화 루프
while True:
    user_input = input("\n사용자: ")
    if user_input == "exit":
        print("Bye")
        break

    # 사용자 메시지를 대화 히스토리에 추가
    messages.append({"role": "user", "content": user_input})

    print("\n에이전트: ", end="", flush=True)

    # 스트리밍 방식으로 에이전트 실행 (전체 대화 히스토리 전달)
    result = Runner.run_streamed(agent, input=messages)

    full_response = ""  # 스트리밍 응답 조각을 누적할 변수

    # 스트리밍 이벤트를 실시간으로 처리
    async for event in result.stream_events():
        # 텍스트 델타 이벤트만 필터링하여 출력
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            print(delta, end="", flush=True)  # 글자 단위로 실시간 출력
            full_response += delta            # 전체 응답 누적

    # 완성된 에이전트 응답을 대화 히스토리에 추가
    messages.append({"role": "assistant", "content": full_response})

# %%

# %% [markdown]
# ### 실습 문제
#
# 본문에서 배운 **스트리밍 + 함수 도구 + RAG**를 모두 조합해 **고객 지원 챗봇**을 완성하세요.
#
# 1. **주문 조회 도구**:
#    `@function_tool get_order_status(order_id: str) -> str` 를 정의하세요.
#    더미 주문 데이터(dict)에서 주문 상태를 조회해 반환하고, 없는 주문 번호면
#    `"주문을 찾을 수 없습니다"` 를 반환하세요.
#
# 2. **RAG 도구**:
#    본문에서 만든 Vector Store(`vs_id`)를 `FileSearchTool`로 연결해
#    환불 정책·비밀번호 등 **FAQ 질문**에 지식 베이스를 검색해 답하도록 하세요.
#
# 3. **스트리밍 대화 루프**:
#    `Runner.run_streamed()` + `ResponseTextDeltaEvent`로 답변을 토큰 단위로 출력하고,
#    `messages` 리스트에 대화 기록을 누적하세요. (`"exit"` 입력 시 종료)
#
# ### 테스트 입력 예시
#
# * `"환불 정책이 어떻게 되나요?"`
#   👉 file_search 도구 → FAQ 검색 후 답변
#
# * `"주문번호 ORD-1001 상태 알려줘"`
#   👉 get_order_status 도구 실행 → 주문 상태 출력
#
# * `"그 주문 언제쯤 도착할까요?"`
#   👉 이전 대화를 기억하는지(멀티턴) 확인
#

# %% [markdown]
# ---------------
# ### 웹 UI 구현으로 연결
#
# 동일한 에이전트 구조(get_weather + WebSearchTool + FileSearchTool)를 **Flask 웹 앱**으로 구현한 예제가 **`01_OpenAI_Agent_Basic/11_rag-chat-flask/app.py`** 입니다.
# Flask API + HTML 한 장(바닐라 JavaScript)으로 채팅 화면, 답변 생성, 대화 기록 유지까지 포함합니다.
#
# 실행: `11_rag-chat-flask` 폴더에서 `python app.py` 실행 후 브라우저에서 http://localhost:8000 접속

# %%
