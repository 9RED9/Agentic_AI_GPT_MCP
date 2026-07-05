"""
LangChain ReAct Agent 기반 Chatbot (Flask + HTML)

- 에이전트 구성은 101_Tools_Agents + 102_Memory_Concepts 노트북 내용을 반영:
  - 고객 DB 검색(search_db) + 현재 기온 조회(get_weather) 도구를 사용하는 ReAct Agent
  - InMemorySaver(checkpointer) 기반 단기 메모리로 대화 이력을 자동 관리
- 실행: python x101_Tools_Memory_Agent.py  ->  브라우저에서 http://localhost:8000 접속

라우트 구성:
  GET  /         채팅 UI(101_Tools_Memory_Agent.html) 반환
  POST /chat     {"message": "..."} 를 받아 에이전트 실행 후 {"reply": "..."} 반환
  POST /reset    대화 기록 초기화 (새 thread_id 발급)
  POST /summary  지금까지의 대화 내용을 LLM으로 요약하여 반환
"""

# ---------------------------------------------------------------------------------
# import (표준 라이브러리 / Flask / LangChain / LangGraph)
# ---------------------------------------------------------------------------------










# ---------------------------------------------------------------------------------
# 환경 변수 로드 및 기본 상수 정의 (MODEL_NAME, PORT, BASE_DIR, SYSTEM_PROMPT)
# ---------------------------------------------------------------------------------






# ---------------------------------------------------------------------------------
# 도구 정의 (101_Tools_Agents 노트북과 동일)
# ---------------------------------------------------------------------------------

# search_db: 검색어(query)와 limit을 받아 고객 DB 검색 결과 문자열 반환 (@tool 데코레이터 사용)






# WeatherInput: 위도(latitude), 경도(longitude)를 담는 Pydantic 입력 스키마




# get_weather: open-meteo API로 현재 기온을 조회 (@tool(args_schema=WeatherInput) 사용)
# https://api.open-meteo.com/v1/forecast?latitude=...&longitude=...&current=temperature_2m












# ---------------------------------------------------------------------------------
# LLM 및 ReAct Agent 초기화 (서버 시작 시 1회만 생성해서 재사용)
# - 102_Memory_Concepts 노트북과 동일하게 InMemorySaver를 checkpointer로 사용
#   -> 같은 thread_id로 호출하면 이전 대화를 자동으로 기억 (단기 메모리)
#   -> 서버 프로세스가 종료되면 대화 이력도 사라짐
# ---------------------------------------------------------------------------------

# llm 생성 (init_chat_model)

# checkpointer 생성 (InMemorySaver)

# agent 생성 (create_agent: llm + 도구 목록 + system_prompt + checkpointer)





# ---------------------------------------------------------------------------------
# 대화 스레드 관리
# - 대화 이력 자체는 checkpointer가 thread_id별로 저장하므로,
#   서버는 "현재 사용 중인 thread_id"만 관리하면 됨
# - 대화 초기화는 새 thread_id를 발급하는 방식으로 구현 (빈 대화에서 새로 시작)
# - threaded=True로 실행하면 요청이 동시에 처리될 수 있으므로 Lock으로 보호
# ---------------------------------------------------------------------------------




# current_config(): 현재 thread_id로 에이전트 호출용 config 생성
#   반환 형식: {"configurable": {"thread_id": "conversation_N"}}





# ---------------------------------------------------------------------------------
# Flask 앱과 라우트
# ---------------------------------------------------------------------------------



# GET / : 채팅 UI 페이지(101_Tools_Memory_Agent.html) 반환 (send_file 사용)





# POST /chat : 사용자 메시지를 받아 에이전트를 실행하고 답변을 JSON으로 반환
#   - request.get_json()에서 "message" 추출, 비어 있으면 400 에러
#   - 이전 대화 이력은 checkpointer가 thread_id 기준으로 자동 관리하므로
#     매 요청마다 새 사용자 메시지 하나만 전달하면 됨
#   - agent.invoke() 결과의 마지막 메시지에서 답변 추출













# POST /reset : 대화 기록 초기화 - 새 thread_id를 발급하여 빈 대화에서 새로 시작






# POST /summary : checkpointer에 저장된 현재 스레드의 대화 이력을 LLM으로 요약하여 반환

    # 저장된 대화 이력 조회 (agent.get_state() 사용, 102 노트북의 '대화 이력 조회'에 해당)



    # 대화가 없으면 "요약할 대화가 없습니다." 반환



    # 메시지 역할별로 텍스트로 변환 (System / User / AI, 도구 호출 등 그 외 메시지는 제외)










    # 요약 프롬프트 생성 후 LLM에게 요약 요청 (llm.invoke)







# ---------------------------------------------------------------------------------
# 서버 실행 (host="127.0.0.1", port=PORT, threaded=True)
# ---------------------------------------------------------------------------------
