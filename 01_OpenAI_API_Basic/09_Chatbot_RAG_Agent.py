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
# # 09. Chatbot RAG Agent 구현

# %% [markdown]
# ## 이 노트북의 실습 구성
#
# 이 노트북에서는 **OpenAI Agents SDK**로 채팅 에이전트를 단계별로 확장합니다. 아래 순서대로 진행합니다.
#
# 1. **비동기 방식 호출**: 도구 없이 Agent + Runner로 대화만 구현 (메시지 유지, `await Runner.run`)
# 2. **Streaming 방식 호출**: 같은 에이전트를 `Runner.run_streamed`로 실행해 토큰 단위 실시간 출력
# 3. **도구를 추가한 Chatbot Agent**: `@function_tool` 날씨 도구 + `WebSearchTool`을 에이전트에 연결
# 4. **RAG(검색 증강 생성) 도구 추가**: `10_chat-app-rag`의 지식 베이스 검색을 도구로 붙여, 계정·환불·정책 등 FAQ 기반 답변
# 5. **웹 UI 구현으로 연결**: 동일 RAG를 Streamlit 웹 앱(`10_chat-app-rag/app.py`)으로 구현한 예제 안내

# %%
import openai

from dotenv import load_dotenv
load_dotenv() 

Model = "gpt-5-nano"

# %% [markdown]
# ### 비동기 방식 호출

# %%
from agents import Agent, Runner  

messages = []  # 지금까지의 대화 내용을 저장할 리스트

agent = Agent(
    model=Model, 
    name="여행 에이전트",
    instructions="당신은 훌륭한 여행 에이전트입니다. 사용자와 대화하면서 여행 계획을 도와주세요."
)


while True:
    # 사용자 입력 받기
    user_input = input("\n사용자: ")
    
    # 'exit' 입력 시 종료
    if user_input == "exit":
        print("Bye")
        break
    
    # 현재 사용자의 발화를 대화 기록에 추가
    messages.append({"role": "user", "content": user_input})

    response = await Runner.run(agent, input=messages)
    
    # 모델의 응답을 대화 기록에 추가 (assistant 역할)
    messages.append({"role": "assistant", "content": response.final_output})
    
    # 모델의 최종 응답 출력
    print(f"\n여행 에이전트: {response.final_output}")

# %% [markdown]
# ### Streaming 방식 호출

# %%
#  모델의 답변이 완성될 때까지 기다리지 않고, 한 토큰씩 실시간으로 출력되는 "스트리밍" 방식
from openai.types.responses import ResponseTextDeltaEvent

messages = []  

agent = Agent(
    model=Model,
    name="여행 에이전트",
    instructions="당신은 훌륭한 여행 에이전트입니다. 사용자와 대화하면서 여행 계획을 도와주세요.",
)

while True:
    # 사용자 입력 받기
    user_input = input("\n사용자: ")
    
    # 종료 조건
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
# **RAG**는 지식 베이스(FAQ, 문서)를 임베딩해 두고, 사용자 질문과 유사한 문서를 검색한 뒤 그 내용을 맥락으로 답변하는 방식입니다.
# 에이전트에 **지식 베이스 검색 도구**를 하나 추가하면, 계정·환불·정책 등 관련 질문에 대해 검색 결과를 바탕으로 답할 수 있습니다.
# 아래에서는 `10_chat-app-rag` 폴더의 `rag` 모듈을 사용해 동일한 검색 로직을 도구로 붙입니다.

# %%
from pathlib import Path
import sys

RAG_DIR = Path.cwd() / "10_chat-app-rag"
sys.path.insert(0, str(RAG_DIR))

from agents import Agent, Runner, function_tool, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent

try:
    from rag import get_retrieved_context
except ImportError:
    get_retrieved_context = None

@function_tool
def search_knowledge_base(query: str) -> str:
    """사용자 질문에 맞는 FAQ/문서 맥락을 검색합니다.
    계정, 비밀번호, 환불, 고객지원, 보안, 회사 정책 관련 질문일 때 이 도구를 사용하세요."""
    if get_retrieved_context is None:
        return "RAG 모듈을 불러올 수 없습니다. 10_chat-app-rag 폴더가 있고 OPENAI_API_KEY가 설정되어 있는지 확인하세요."
    return get_retrieved_context(query, "openai")


messages = []

tools_list = [get_weather, WebSearchTool()]
if get_retrieved_context is not None:
    tools_list.append(search_knowledge_base)

print(tools_list)

agent = Agent(
    model=Model,
    name="여행·지식 베이스 에이전트",
    instructions=(
        "당신은 여행 계획과 회사 FAQ를 도와주는 에이전트입니다. "
        "날씨가 필요하면 get_weather, 최신 정보는 WebSearchTool, "
        "계정·환불·정책 등은 search_knowledge_base로 지식 베이스를 검색한 뒤 답하세요. "
        "답은 간결하게 하세요."
    ),
    tools=tools_list,
)

while True:
    user_input = input("\n사용자: ")

    if user_input == "exit":
        print("Bye")
        break

    messages.append({"role": "user", "content": user_input})

    print("\n에이전트: ", end="", flush=True)

    result = Runner.run_streamed(agent, input=messages)
    full_response = ""

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            delta = event.data.delta or ""
            print(delta, end="", flush=True)
            full_response += delta

    messages.append({"role": "assistant", "content": full_response})

# %% [markdown]
# ---------------
# ### 웹 UI 구현으로 연결
#
# 위에서 사용한 **RAG 도구**와 동일한 방식을 **Streamlit 웹 앱**으로 구현한 예제가 **`01_OpenAI_API_Basic/10_chat-app-rag/app.py`** 입니다.
# 해당 앱에서는 채팅 화면, OpenAI로 답변 생성, 대화 기록 유지까지 포함하므로,
# 노트북에서 익힌 RAG 도구가 웹에서 어떻게 쓰이는지 확인할 수 있습니다.
#
# 실행: `10_chat-app-rag` 폴더에서 `streamlit run app.py`

# %%
