# Runner.run_sync() 동기 실행 예제
#
# Jupyter에서는 이미 이벤트 루프가 실행 중이므로 run_sync()가 에러를 발생시킵니다.
# ("AgentRunner.run_sync() cannot be called when an event loop is already running")
#
# 이 파일처럼 일반 Python 스크립트로 실행하면 실행 중인 이벤트 루프가 없으므로
# run_sync()가 내부적으로 새 이벤트 루프를 만들어 정상 동작합니다.
#
# 실행 방법:
#   python 02_agent_run_sync.py

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, function_tool

Model = "gpt-5.4-mini"


# 직사각형의 넓이를 계산하는 도구
@function_tool
def calculate_area(length: float, width: float) -> float:
    """직사각형의 가로(length)와 세로(width)를 받아 넓이를 반환한다."""
    print(f"** calculate_area 함수 실행 ** 가로: {length}, 세로: {width}")
    return length * width


agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요. 넓이 계산은 반드시 calculate_area 도구를 사용하세요.",
    model=Model,
    tools=[calculate_area],
)

# 1. 단순 질문 - 동기 실행 (await 불필요)
result = Runner.run_sync(agent, "2+2는 얼마일까요? 한 단어로 답하세요.")
print("답변 1:", result.final_output)

print("-" * 60)

# 2. 도구 호출을 포함한 질문 - run_sync도 에이전트 루프(도구 실행)를 자동 처리
result = Runner.run_sync(agent, "가로 5, 세로 7인 직사각형의 넓이를 구해주세요.")
print("답변 2:", result.final_output)
