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
# # Deep Agent (딥 에이전트)
#
# **Deep Agent**는 데이터 분석처럼 계획(planning), 코드 실행(code execution),
# 아티팩트(스크립트, 리포트, 시각화 등) 생성이 필요한 복잡한 작업을 처리하기 위해 설계된 에이전트입니다.
#
# 이 노트북에서는 Deep Agent를 사용하여:
# 1. CSV 파일을 분석하고
# 2. 커스텀 도구를 통해 결과를 공유하는
#
# 데이터 분석 에이전트를 구축합니다.
#
# **참고**:
# - [LangChain 공식 문서 - Deep Agents](https://docs.langchain.com/oss/python/deepagents)
# - [LangChain 공식 문서 - Data Analysis](https://docs.langchain.com/oss/python/deepagents/data-analysis)
# - [LangChain 공식 문서 - Backends](https://docs.langchain.com/oss/python/deepagents/backends)
#
# **핵심 개념**:
# - **Backend**: 샌드박스 환경에서 코드를 실행하는 백엔드 시스템
# - **Custom Tools**: 외부 서비스(Slack, 이메일 등)와 연동하는 커스텀 도구
# - **Checkpointer**: 멀티턴 대화를 지원하는 체크포인터

# %% [markdown]
# ## Deep Agent 아키텍처 이해
#
# Deep Agent는 일반 에이전트와 다음과 같은 차이가 있습니다:
#
# ### 일반 에이전트 vs Deep Agent
#
# | 특성 | 일반 에이전트 | Deep Agent |
# |------|-------------|------------|
# | **코드 실행** | 도구를 통한 간접 실행 | 백엔드에서 직접 실행 |
# | **파일 생성** | 제한적 | 스크립트, 리포트, 시각화 등 자유롭게 생성 |
# | **샌드박스** | 없음 | 격리된 환경에서 안전하게 실행 |
# | **계획 수립** | 단순 반복 | 복잡한 작업의 계획과 실행 |
# | **아티팩트** | 텍스트 응답 위주 | 파일, 이미지, 코드 등 다양한 아티팩트 생성 |
#
# ### 실행 흐름
#
# ```
# 사용자 요청
#     ↓
# Deep Agent (계획 수립)
#     ↓
# Backend (코드 실행)
#     ↓
# 아티팩트 생성 (스크립트, 차트, 리포트)
#     ↓
# 커스텀 도구 (결과 공유)
#     ↓
# 사용자에게 응답
# ```

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %% [markdown]
# ## 1. 설치
#
# Deep Agent를 사용하려면 `deepagents` 패키지를 설치해야 합니다.
#
# ```bash
# pip install deepagents
# ```
#
# 선택적 의존성 (Slack 연동 시):
# ```bash
# pip install slack-sdk
# ```

# %%
# # !pip install deepagents

# %% [markdown]
# ## 2. Backend 설정
#
# Deep Agent는 **Backend**를 통해 샌드박스 환경에서 코드를 실행합니다.
# 여러 백엔드 제공자를 사용할 수 있습니다:
#
# | 백엔드 | 설명 | 보안 |
# |--------|------|------|
# | **Daytona** | 클라우드 샌드박스 | 높음 |
# | **Modal** | 서버리스 샌드박스 | 높음 |
# | **Runloop** | 클라우드 개발 환경 | 높음 |
# | **LocalShell** | 로컬 셸 (개발/테스트용) | 낮음 (주의 필요) |
#
# > **주의**: `LocalShellBackend`는 파일시스템과 셸에 대한 제한 없는 접근을 제공합니다.
# > 개발/테스트 환경에서만 사용하세요. 프로덕션에서는 샌드박스 백엔드를 권장합니다.
#
# > **Windows 참고**: `LocalShellBackend`의 내장 파일 도구(`read_file`, `ls`)는 Windows 절대 경로를
# > 지원하지 않습니다. Windows에서는 에이전트가 `execute` 도구로 Python 명령을 실행하여
# > 파일을 읽도록 프롬프트에서 안내하는 것이 좋습니다.

# %%
from deepagents.backends import LocalShellBackend

# 작업 디렉토리 절대 경로
WORK_DIR = os.getcwd()

# 로컬 셸 백엔드 설정 (개발/테스트용)
# root_dir에 절대 경로를 지정하면 execute 명령이 이 디렉토리에서 실행됩니다
import locale

# 서브프로세스 출력은 부모 프로세스(주피터 커널)의 기본 인코딩으로 디코딩되므로,
# 자식 python의 출력 인코딩을 부모와 일치시킵니다 (한글 Windows: cp949).
# ':replace' — 인코딩할 수 없는 문자는 오류 대신 대체 문자로 처리
SUBPROCESS_ENC = locale.getpreferredencoding(False)

backend = LocalShellBackend(
    root_dir=WORK_DIR,
    env={
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),  # execute 도구가 python 명령을 찾는 데 필요
        "PYTHONIOENCODING": f"{SUBPROCESS_ENC}:replace",
    },
)

# 백엔드 동작 확인
result = backend.execute("echo ready")
print(result)
print(f"작업 디렉토리: {WORK_DIR}")

# %% [markdown]
# ## 3. 샘플 데이터 준비
#
# 분석할 CSV 데이터를 생성하고 백엔드에 업로드합니다.

# %%
import csv
import io
# 샘플 판매 데이터 생성
data = [
    ["Date", "Product", "Units Sold", "Revenue"],
    ["2025-08-01", "사과", 10, 250],
    ["2025-08-02", "바나나", 5, 125],
    ["2025-08-03", "사과", 7, 175],
    ["2025-08-04", "체리", 3, 90],
    ["2025-08-05", "바나나", 8, 200],
    ["2025-08-06", "사과", 12, 300],
    ["2025-08-07", "체리", 6, 180],
    ["2025-08-08", "바나나", 9, 225],
    ["2025-08-09", "사과", 15, 375],
    ["2025-08-10", "체리", 4, 120],
]
# CSV 파일을 로컬 파일시스템에 직접 저장
data_dir = os.path.join(WORK_DIR, "data")
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, "sales_data.csv")

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerows(data)
print("데이터 저장 완료!")

print(f"파일 경로: {csv_path}")
print(f"행 수: {len(data) - 1}")  # 헤더 제외
print(f"컬럼: {data[0]}")
# 파일 내용 확인
import pandas as pd
df_check = pd.read_csv(csv_path)
df_check.head()

# %% [markdown]
# ## 4. 커스텀 도구 구현
#
# Deep Agent에게 추가 기능을 제공하는 커스텀 도구를 만듭니다.
# 에이전트가 생성한 아티팩트(리포트, 시각화 등)를 외부로 공유할 수 있습니다.
#
# 여기서는 **로컬 파일 저장** 도구를 만들어 결과를 저장합니다.

# %%
from langchain.tools import tool

@tool
def save_report(text: str, file_path: str | None = None) -> str:
    """분석 결과를 로컬 파일에 저장합니다.

    Args:
        text: 저장할 텍스트 내용 (분석 리포트)
        file_path: 저장할 파일 경로. 지정하지 않으면 기본 경로에 저장됩니다.
    """
    if not file_path:
        file_path = os.path.join(WORK_DIR, "data", "analysis_report.txt")

    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return f"리포트가 '{file_path}'에 저장되었습니다."

@tool
def download_artifact(file_path: str) -> str:
    """백엔드에서 생성된 아티팩트(이미지, 파일 등)를 다운로드합니다.
    Args:
        file_path: 다운로드할 파일의 경로
    """
    # 백엔드에서 지정된 경로의 파일을 다운로드 (리스트로 전달)
    files = backend.download_files([file_path])
    
    # 파일이 존재하면 성공 메시지와 파일 크기(bytes) 반환
    if files:
        return f"'{file_path}' 파일을 성공적으로 다운로드했습니다. ({len(files[0].content)} bytes)"
    
    # 파일을 찾지 못한 경우 실패 메시지 반환
    return f"'{file_path}' 파일을 찾을 수 없습니다."

print("커스텀 도구 생성 완료:")
print(f"  - save_report: {save_report.description[:50]}...")
print(f"  - download_artifact: {download_artifact.description[:50]}...")

# %% [markdown]
# ## 5. Deep Agent 생성 및 실행
#
# `create_deep_agent()` 팩토리 함수로 에이전트를 생성합니다.
#
# ### 내장 도구 (기본 제공)
#
# create_deep_agent는 별도 등록 없이 다음 도구들을 자동으로 포함합니다.
# 아래 실행 출력에서 에이전트가 이 도구들을 호출하는 것을 볼 수 있습니다.
#
# | 내장 도구 | 역할 |
# |---|---|
# | `write_todos` | 작업 계획 수립/추적 (복잡한 작업을 단계로 분해) |
# | `ls`, `read_file`, `write_file`, `edit_file` | 백엔드 파일시스템 탐색/읽기/쓰기/수정 |
# | `task` | 서브에이전트 호출 (독립 컨텍스트로 하위 작업 위임) |
# | `execute` | 셸 명령/코드 실행 (Sandbox 백엔드 연결 시) |
#
# 여기에 `tools` 파라미터로 **확장 도구**(웹 검색, MCP 어댑터, 커스텀 도구 등)를 추가할 수 있습니다.
#
# ### 주요 파라미터
#
# - **model**: 사용할 LLM 모델
# - **tools**: 커스텀 도구 목록
# - **backend**: 코드 실행 백엔드 (기본: 인메모리 상태 기반, 교체 가능)
# - **system_prompt**: 에이전트 역할/지침
# - **checkpointer**: 멀티턴 대화 지원용 체크포인터
# - **subagents / skills / memory / interrupt_on**: 고급 기능 (→ [401_Deep_Agent_Advanced.ipynb](401_Deep_Agent_Advanced.ipynb))
#

# %%
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

# 체크포인터 생성 - 멀티턴 대화의 상태를 메모리에 저장
checkpointer = InMemorySaver()

# Deep Agent 생성
agent = create_deep_agent(
    model="openai:gpt-5-mini",       # 사용할 LLM 모델
    tools=[save_report, download_artifact],  # 커스텀 도구 등록
    backend=backend,                  # 코드 실행 백엔드 (LocalShell)
    system_prompt=(                   # 에이전트 역할/지침
        "당신은 데이터 분석 전문가입니다. "
        "계획을 세운 뒤 코드를 실행해 데이터를 분석하고, 결과를 리포트로 정리하세요."
    ),
    checkpointer=checkpointer,        # 대화 상태 관리용 체크포인터
)

# 고유한 스레드 ID 생성 - 대화 세션을 식별하는 데 사용
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

print(f"Deep Agent 생성 완료 (thread_id: {thread_id})")

# %% [markdown]
# ### 에이전트에게 분석 요청
#
# 자연어로 데이터 분석을 요청하면, 에이전트가 자동으로:
# 1. 데이터를 읽고 탐색
# 2. 분석 코드를 작성하고 실행
# 3. 커스텀 도구를 사용하여 결과를 공유

# %%
# NOTE: LocalShellBackend의 read_file/ls 도구는 Windows 경로를 지원하지 않으므로,
# 에이전트가 execute 도구로 Python/pandas를 사용해 파일을 읽도록 안내합니다.

# Windows 백슬래시(\)를 슬래시(/)로 변환 - Agent가 경로를 올바르게 인식하도록
csv_path_for_agent = csv_path.replace("\\", "/")

# Agent에게 전달할 사용자 메시지 구성
# - CSV 파일 경로와 분석 지시사항을 포함
# - read_file/ls 대신 execute 도구 사용을 명시 (Windows 호환성)
input_message = {
    "role": "user",
    "content": (
        f"'{csv_path_for_agent}' 경로에 있는 CSV 파일을 분석해주세요. "
        "반드시 execute 도구를 사용하여 Python(pandas)으로 데이터를 읽고 분석하세요. "
        "데이터가 한달치 이므로 일별 매출 추세를 분석하세요."
        "레포트는 한국어로 작성하세요."
        "read_file이나 ls 도구는 사용하지 마세요. "
        "분석이 끝나면 save_report 도구를 사용하되, "
        "file_path 인자 없이 호출하여 기본 경로에 리포트를 저장해주세요."
    ),
}

# Agent 실행 - 스트리밍 모드로 각 단계의 업데이트를 실시간 출력
for step in agent.stream(
    {"messages": [input_message]},  # 사용자 메시지를 입력으로 전달
    config,                         # thread_id가 포함된 설정
    stream_mode="updates",          # 각 노드의 업데이트만 스트리밍
):
    # 각 스텝에서 메시지 추출 후 출력
    for _, update in step.items():
        if update and (messages := update.get("messages")) and isinstance(messages, list):
            for message in messages:
                message.pretty_print()  # LangChain 메시지 포맷으로 보기 좋게 출력

# %% [markdown]
# ## 6. 멀티턴 대화
#
# 체크포인터 덕분에 동일한 `thread_id`로 후속 질문을 할 수 있습니다.
# 에이전트는 이전 대화 맥락을 기억합니다.

# %%
# 후속 질문 메시지 구성
# - 이전 대화 컨텍스트(체크포인터)를 활용한 멀티턴 대화
followup_message = {
    "role": "user",
    "content": (
        "가장 잘 팔린 제품은 무엇인가요? 그리고 일별 매출 추세를 분석해주세요. "
        "execute 도구로 Python 코드를 실행하여 분석하세요."
        "레포트는 한국어로 작성하세요."
    ),
}

# 후속 질문 실행 - 동일한 config(thread_id)로 대화 이어가기
for step in agent.stream(
    {"messages": [followup_message]},  # 후속 질문 전달
    config,                            # 동일한 thread_id로 대화 연속성 유지
    stream_mode="updates",
):
    # 각 스텝에서 메시지 추출 후 출력
    for _, update in step.items():
        if update and (messages := update.get("messages")) and isinstance(messages, list):
            for message in messages:
                message.pretty_print()

# %% [markdown]
# ## 7. 아티팩트 확인
#
# 에이전트가 생성한 파일(시각화 차트, 리포트 등)을 확인합니다.

# %%
# === 저장된 리포트 파일 확인 ===
default_report_path = os.path.join(WORK_DIR, "data", "analysis_report.txt")

if os.path.exists(default_report_path):
    # 리포트 파일이 존재하면 내용 출력
    with open(default_report_path, "r", encoding="utf-8") as f:
        print("=== 저장된 분석 리포트 ===")
        print(f.read())
else:
    print(f"리포트 파일이 아직 생성되지 않았습니다: {default_report_path}")


# %% [markdown]
#

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **Deep Agent**: 계획, 코드 실행, 아티팩트 생성이 가능한 고급 에이전트
# 2. **Backend**: 샌드박스 환경에서 안전하게 코드를 실행 (Daytona, Modal, Runloop, LocalShell)
# 3. **Custom Tools**: 외부 서비스와 연동하는 도구를 만들어 결과를 공유
# 4. **Checkpointer**: 멀티턴 대화를 지원하여 이전 맥락을 유지
# 5. **Artifacts**: 스크립트, 리포트, 시각화 등 다양한 아티팩트를 생성하고 관리
#
# **다음 단계**:
# - [401_Deep_Agent_Advanced.ipynb](401_Deep_Agent_Advanced.ipynb): Human-in-the-Loop, Skills, Memory 고급 기능 실습
# - [Deep Agents Customization](https://docs.langchain.com/oss/python/deepagents/customization): 모델, 도구, 프롬프트, 계획 전략 커스터마이징
# - [Deep Agents Sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes): 샌드박스 제공자별 설정 가이드
# - [Deep Agents Skills](https://docs.langchain.com/oss/python/deepagents/skills): 재사용 가능한 스킬 구현

# %%

# %% [markdown]
# ## 실습문제: 나만의 데이터 분석 에이전트
#
# 본문의 흐름(데이터 준비 → 커스텀 도구 → Deep Agent 생성 → 분석 요청 → 멀티턴)을 새로운 데이터에 그대로 적용해 보세요.
#
# 1. **데이터 준비**: 도서 판매 데이터 `data/book_sales.csv`를 만드세요.
#    - 컬럼: Date, Genre, Units Sold, Revenue — 장르 3종(소설/에세이/과학) 8행 정도면 충분합니다.
# 2. **커스텀 도구**: 요약 텍스트를 `data/book_summary.txt`에 저장하는 `save_summary` 도구를 만드세요. (본문 `save_report`와 같은 구조)
# 3. **에이전트 생성 + 분석**: `create_deep_agent`(model, tools, backend, system_prompt, checkpointer)로 에이전트를 만들고 **장르별 매출 분석**을 요청하세요.
#    - 본문처럼 execute 도구로 pandas를 사용하도록 지시하고, 경로는 슬래시(/)로 변환해 전달하세요.
# 4. **멀티턴**: 같은 thread_id로 후속 질문(예: "가장 매출이 높은 장르는?")을 던져 이전 분석 맥락이 유지되는지 확인하세요.
#

# %%
