#---------------------------------------------------------
# Streamlit Session 기반 Chatbot 구현 - LangChain 방식
# Part 1 (Models & Messages) + Part 2 (Context Engineering) 통합
#---------------------------------------------------------
# .env 파일에서 환경 변수를 읽어옵니다.
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# streamlit_chat 라이브러리
from streamlit_chat import message  # 채팅 말풍선 형태로 메시지를 보여줍니다.

# LLM 초기화 (init_chat_model 방식)
llm = init_chat_model("gpt-5-nano", model_provider="openai")

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
# 웹 애플리케이션의 페이지 제목과 아이콘을 설정
st.set_page_config(page_title="나만의 ChatGPT", page_icon=":robot_face:")

# 페이지 제목을 중앙에 정렬하여 표시
# - unsafe_allow_html=True: HTML을 직접 사용할 수 있도록 허용
st.markdown("<h1 style='text-align: center;'>우리 즐겁게 대화해요</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 사이드바: Context Engineering 기능 추가
# ---------------------------------------------------------------------------------
st.sidebar.title("😎 설정")

# Part 2: 시스템 프롬프트 커스터마이징
st.sidebar.subheader("시스템 프롬프트 설정")
st.sidebar.markdown("**Context Engineering** - 역할과 행동 방식을 정의합니다")

# 프롬프트 템플릿 선택
prompt_template = st.sidebar.selectbox(
    "프롬프트 템플릿 선택",
    [
        "기본 (친구처럼)",
        "데이터 분석 전문가",
        "코딩 어시스턴트",
        "번역가",
        "요약 전문가"
    ]
)

# 템플릿별 시스템 프롬프트
prompt_templates = {
    "기본 (친구처럼)": "당신은 친구처럼 말합니다. 모든 질문에 최선을 다해 대답하세요.",
    "데이터 분석 전문가": """당신은 데이터 분석 전문가입니다.
- 데이터 분석 결과를 간결하고 명확하게 설명합니다.
- 전문 용어는 한국어로 번역하여 사용합니다.
- 3-5개의 핵심 포인트로 요약합니다.""",
    "코딩 어시스턴트": """당신은 웹 프레임워크에 전문성을 가진 시니어 Python 개발자입니다.
- 설명은 간결하게 5줄 이내로 설명하세요.
- 코드 예제를 포함하여 설명하세요.""",
    "번역가": "다음을 영어에서 한국어로 번역하세요. 상세한 설명 말고 단순히 번역만 하세요.",
    "요약 전문가": """당신은 요약 전문가입니다.
- 핵심 내용만 간결하게 요약합니다.
- 불필요한 세부사항은 제외합니다.
- 3-5개의 핵심 포인트로 정리합니다."""
}

# 선택된 템플릿에 따라 시스템 프롬프트 설정
system_prompt_text = prompt_templates[prompt_template]
st.sidebar.info(f"**현재 프롬프트:**\n{system_prompt_text}")

# 버튼
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

# ---------------------------------------------------------------------------------
# Streamlit Session State 초기화 (최초 1회만)
# ---------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt_text)
    ]
    st.session_state.current_prompt = system_prompt_text

# 프롬프트가 변경되면 업데이트
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = system_prompt_text

if st.session_state.current_prompt != system_prompt_text:
    # 첫 번째 메시지가 SystemMessage인 경우 업데이트
    if st.session_state.messages and isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages[0] = SystemMessage(content=system_prompt_text)
    else:
        # SystemMessage가 없으면 맨 앞에 추가
        st.session_state.messages.insert(0, SystemMessage(content=system_prompt_text))
    st.session_state.current_prompt = system_prompt_text
    st.sidebar.success("시스템 프롬프트가 업데이트되었습니다!")

# ---------------------------------------------------------------------------------
# 사이드바 버튼 동작 정의
# ---------------------------------------------------------------------------------
# 1) "대화 내용 초기화" 버튼: 대화 기록 리셋
if refresh_button:
    st.session_state.messages = [
        SystemMessage(content=system_prompt_text)
    ]
    st.session_state.current_prompt = system_prompt_text
    st.sidebar.success("대화가 초기화되었습니다!")

# 2) "대화 내용 요약" 버튼: LLM에게 요약을 요청해 결과를 사이드바에 표시
if summaries_button:
    
    # 2-1) 메시지들을 텍스트로 합치기
    conversation_text = []
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        conversation_text.append(f"{role}: {msg.content}")
        
    joined_conversation = "\n".join(conversation_text)

    # 2-2) 요약 프롬프트 만들기 (Part 2: 명확한 지시사항 제공 예제)
    prompt_content = f"""다음 대화를 요약해주세요:
{joined_conversation}
--- 
요약:"""

    # 2-3) LLM에게 요약 요청
    summary_response = llm.invoke([HumanMessage(content=prompt_content)])
    summary_text = summary_response.content

    # 2-4) 사이드바에 요약 결과 표시
    st.sidebar.write("**대화 요약:**")
    st.sidebar.write(summary_text)

# ---------------------------------------------------------------------------------
# 메인 영역: 입력 폼 및 모델 호출
# clear_on_submit=True : 폼이 제출될 때 입력 필드 자동 초기화
# ---------------------------------------------------------------------------------
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # 사용자 입력을 HumanMessage 로 추가
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # LLM을 직접 호출해 AI 응답 생성 (Streamlit Session 방식)
        try:
            response = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as e:
            error_msg = f"에러가 발생했습니다: {str(e)}"
            st.session_state.messages.append(AIMessage(content=error_msg))

# ---------------------------------------------------------------------------------
# "마지막 AIMessage" 폼 바로 아래에 표시
# ---------------------------------------------------------------------------------
# - 대화 이력이 비어있지 않고, 마지막 메시지가 AIMessage이면 표시
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, AIMessage):
        st.text(last_msg.content)

# ---------------------------------------------------------------------------------
# 그 외의 대화 이력(마지막 메시지 제외)을 아래에서 순서대로 표시
# is_user=True : 사용자가 입력한 메세지
# ---------------------------------------------------------------------------------
st.subheader("이전 대화 이력")
for idx, msg in enumerate(st.session_state.messages):  # 맨 마지막 AIMessage는 제외
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=str(idx) + "_user")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=str(idx) + "_ai")
