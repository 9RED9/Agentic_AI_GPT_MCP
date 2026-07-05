import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()

# --------------------------------------------------------
# 1️⃣ 모델 초기화
# --------------------------------------------------------
model = init_chat_model("gpt-5-mini", model_provider="openai")
# model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# --------------------------------------------------------
# 2️⃣ MultiServerMCPClient 설정
# --------------------------------------------------------
client = MultiServerMCPClient(
    {
        # Chinook Database MCP 서버 (stdio 기반 로컬 실행)
        "chinook": {
            "transport": "stdio",       # 로컬 subprocess 통신
            "command": "python",        # MCP 서버 실행 명령어
            "args": ["./agent_server.py"],  # MCP 서버 스크립트 경로
        }
    }
)

# --------------------------------------------------------
# 3️⃣ 메인 실행 함수
# --------------------------------------------------------
async def run():
    print("===== MCP MultiServer 클라이언트 초기화 중... =====")

    # MCP 서버 연결 및 도구 목록 가져오기
    tools = await client.get_tools()

    # 메모리 체크포인터 생성 (대화 히스토리 유지용)
    memory = MemorySaver()

    # LangChain ReAct 에이전트 생성
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )

    print("\n===== Chinook 데이터베이스 대화형 챗봇 시작 =====")
    print("대화 히스토리가 메모리에 저장됩니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    session_id = "user_session_1"

    # --------------------------------------------------------
    # 대화 루프
    # --------------------------------------------------------
    while True:
        user_input = input("질문을 입력하세요: ")

        if user_input.lower() in ["quit", "exit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        user_message = HumanMessage(content=user_input)
        config = {"configurable": {"thread_id": session_id}}

        # MCP 기반 도구 포함한 LangChain 에이전트 호출
        response = await agent.ainvoke(
            {"messages": [user_message]},
            config=config
        )

        print("===== RESPONSE =====")
        print(response["messages"][-1].content)
        print("=" * 50 + "\n")


# --------------------------------------------------------
# 4️⃣ 비동기 루프 실행
# --------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run())