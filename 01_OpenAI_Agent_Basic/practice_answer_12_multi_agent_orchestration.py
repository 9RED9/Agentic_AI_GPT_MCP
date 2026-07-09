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
# # 실습 문제 모범답안 - 12. Multi-Agent Orchestration
#
# `12_multi_agent_orchestration` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# **뉴스 기사 브리핑 파이프라인 (병렬 분석 + 순차 종합 패턴)**
#
# 1. **병렬 분석**: 제목 생성기 / 핵심 문장 추출기 / 카테고리 분류기를 `asyncio.gather()`로 동시 실행
# 2. **순차 종합**: 브리핑 작성기가 세 분석 결과를 종합해 3~4줄 브리핑 작성
# 3. 전체 파이프라인을 `trace()`로 감싸기
#

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import asyncio

Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 병렬 분석 에이전트 3개 정의
#
# 세 에이전트는 서로의 결과가 필요 없는 **독립적인 분석**이므로 병렬 실행 대상입니다.
#

# %%
from agents import Agent, Runner, trace

# 병렬 분석 에이전트 1: 제목 생성기
title_agent = Agent(
    name="제목 생성기",
    instructions="""주어진 뉴스 기사에 어울리는 제목 1개를 만드세요.
결과 형식: "제목: [제목]" """,
    model=Model,
)

# 병렬 분석 에이전트 2: 핵심 문장 추출기
key_sentence_agent = Agent(
    name="핵심 문장 추출기",
    instructions="""주어진 뉴스 기사에서 가장 중요한 문장 2개를 추출하세요.
결과 형식: "핵심 문장: 1) [문장1] 2) [문장2]" """,
    model=Model,
)

# 병렬 분석 에이전트 3: 카테고리 분류기
category_agent = Agent(
    name="카테고리 분류기",
    instructions="""주어진 뉴스 기사를 경제 / 기술 / 사회 중 하나로 분류하세요.
결과 형식: "카테고리: [분류], 이유: [간단한 설명]" """,
    model=Model,
)

# 순차 종합 에이전트: 브리핑 작성기
briefing_agent = Agent(
    name="브리핑 작성기",
    instructions="""당신은 뉴스 브리핑 작성자입니다.
주어진 분석 결과들을 종합하여 3~4줄의 간결한 브리핑을 작성하세요.
브리핑 형식:
---
[뉴스 브리핑]
- 제목: ...
- 카테고리: ...
- 핵심 내용: ...
- 한 줄 평: ...
---""",
    model=Model,
)

# %% [markdown]
# ## 2. 병렬 실행 → 순차 종합
#
# - **Step 1**: 독립적인 세 분석을 `asyncio.gather()`로 동시 실행 (시간 절약)
# - **Step 2**: 세 결과를 하나의 입력으로 묶어 브리핑 작성기에 **순차 전달**
#

# %%
news_article = """
국내 연구진이 차세대 이차전지인 전고체 배터리의 수명을 3배 늘리는
신소재를 개발했다. 이번 기술은 전기차 주행거리 향상에 크게 기여할 것으로
기대되며, 연구팀은 3년 내 상용화를 목표로 국내 배터리 기업들과
협력을 논의 중이라고 밝혔다.
"""

with trace("뉴스 브리핑 파이프라인"):
    # Step 1: 병렬 분석 - 세 에이전트를 동시에 실행
    print("Step 1: 병렬 분석 실행 중...\n")
    title_result, key_result, category_result = await asyncio.gather(
        Runner.run(title_agent, news_article),
        Runner.run(key_sentence_agent, news_article),
        Runner.run(category_agent, news_article),
    )

    print(f"  {title_result.final_output}")
    print(f"  {key_result.final_output}")
    print(f"  {category_result.final_output}")

    # Step 2: 순차 종합 - 세 분석 결과를 브리핑 작성기에 전달
    print("\nStep 2: 브리핑 작성 중...\n")
    combined_input = f"""다음 분석 결과를 종합해 브리핑을 작성해주세요:

원본 기사: {news_article}

분석 결과:
1. {title_result.final_output}
2. {key_result.final_output}
3. {category_result.final_output}
"""
    briefing_result = await Runner.run(briefing_agent, combined_input)

print(f"최종 브리핑:\n{briefing_result.final_output}")

# %% [markdown]
# ### 정리
#
# | 단계 | 패턴 | 코드 |
# |------|------|------|
# | Step 1 | 병렬 실행 | `asyncio.gather(Runner.run(...), Runner.run(...), Runner.run(...))` |
# | Step 2 | 순차 체이닝 | 세 결과를 문자열로 합쳐 `Runner.run(briefing_agent, combined_input)` |
# | 전체 | 추적 | `with trace("뉴스 브리핑 파이프라인"):` 로 한 트레이스에 묶음 |
#
# - 제목/핵심 문장/카테고리는 **서로 독립적**이므로 병렬 실행 → 실행 시간 절약
# - 브리핑 작성은 세 결과가 **모두 필요**하므로 병렬 분석이 끝난 뒤 순차 실행
# - 실전 파이프라인은 이렇게 "독립적인 부분은 병렬, 의존적인 부분은 순차"로 조합합니다.
#
