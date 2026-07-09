from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()

Model = "gpt-5.4-mini"

agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model
)

try:
    result = Runner.run_sync(agent, "2+2는 얼마일까요? 한 단어로 답하세요.")
    print(result.final_output)
except Exception as e:
    print(e)