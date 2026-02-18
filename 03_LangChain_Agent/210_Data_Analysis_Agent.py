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
# # Data Analysis Agent (데이터 분석 에이전트)
#
# 이 튜토리얼에서는 LangChain을 사용하여 CSV 데이터를 분석하는 에이전트를 구축합니다.
# 에이전트는 Pandas DataFrame을 조작하고, 인사이트를 제공할 수 있습니다.
#
# **참고**: [LangChain 공식 문서 - Deep Agents Data Analysis](https://docs.langchain.com/oss/python/deepagents/data-analysis)
#
# ## 학습 내용
# - CSV 파일 로드 및 DataFrame 생성
# - Pandas 도구를 사용한 데이터 분석
# - 분석 리포트 생성 및 인사이트 도출

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith 추적 (선택적)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %%
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 모델 초기화
model = init_chat_model("gpt-5-nano", model_provider="openai")

# %% [markdown]
# ## 1. 데이터 준비
#
# CSV 파일을 로드하여 DataFrame을 생성합니다.

# %%
# 예제 데이터 생성 (실제로는 CSV 파일을 로드)
data = {
    '날짜': pd.date_range('2024-01-01', periods=30, freq='D'),
    '매출': [1000 + i*50 + (i%7)*100 for i in range(30)],
    '비용': [500 + i*20 + (i%5)*50 for i in range(30)],
    '제품': ['A', 'B', 'C'] * 10,
    '지역': ['서울', '부산', '대구'] * 10
}

df = pd.DataFrame(data)
df['이익'] = df['매출'] - df['비용']
df['이익률'] = (df['이익'] / df['매출'] * 100).round(2)

print("데이터 미리보기:")
print(f"\n데이터 크기: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Pandas DataFrame 에이전트 생성
#
# `create_agent`와 pandas 도구를 사용하여 DataFrame을 조작할 수 있는 에이전트를 생성합니다.

# %%
# DataFrame을 전역 변수로 저장 (도구에서 접근 가능하도록)
_global_df = df

# Pandas 작업을 수행하는 도구 생성
@tool
def pandas_query(query: str) -> str:
    """Pandas DataFrame에 대한 쿼리를 실행합니다.
    
    이 도구는 pandas DataFrame을 조작하고 분석하는 데 사용됩니다.
    
    DataFrame 정보:
    - 컬럼: {list(_global_df.columns)}
    - 행 수: {len(_global_df)}
    
    사용 가능한 작업:
    - 데이터 조회: df.head(), df.tail(), df.describe()
    - 필터링: df[df['컬럼'] > 값]
    - 집계: df.groupby('컬럼').mean(), df.groupby('컬럼').sum()
    - 통계: df['컬럼'].mean(), df['컬럼'].max(), df['컬럼'].min()
    - 정렬: df.sort_values('컬럼')
    
    Args:
        query: 실행할 pandas 코드 (df 변수 사용)
               반드시 유효한 Python 코드여야 합니다.
    
    Returns:
        쿼리 실행 결과 (문자열)
    """
    try:
        # 안전한 실행을 위한 제한된 네임스페이스
        safe_dict = {
            "df": _global_df,
            "pd": pd,
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "mean": lambda x: x.mean() if hasattr(x, 'mean') else None,
        }
        
        result = eval(query, {"__builtins__": {}}, safe_dict)
        
        # 결과를 문자열로 변환
        if isinstance(result, pd.DataFrame):
            if len(result) > 20:
                return f"결과 (상위 20행):\n{result.head(20).to_string()}\n... (총 {len(result)}행)"
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except SyntaxError as e:
        return f"구문 오류: {str(e)}\n쿼리를 다시 확인해주세요. 예: df.head() 또는 df.groupby('제품')['매출'].sum()"
    except Exception as e:
        return f"오류 발생: {str(e)}\n사용 가능한 컬럼: {list(_global_df.columns)}"

# 에이전트 생성
df_info_single = f"""
DataFrame 구조:
- 컬럼: {list(_global_df.columns)}
- 행 수: {len(_global_df)}
- 샘플 데이터:
{_global_df.head(3).to_string()}
"""

system_prompt = f"""당신은 데이터 분석 전문가입니다.

{df_info_single}

사용자의 질문에 답하기 위해 pandas_query 도구를 사용하여 DataFrame을 분석하세요.
결과를 명확하고 이해하기 쉽게 설명하세요.
도구를 사용할 때는 정확한 Python 코드를 생성하세요."""

agent = create_agent(
    model,
    tools=[pandas_query],
    system_prompt=system_prompt
)

agent

# %% [markdown]
# ## 3. 데이터 분석 쿼리
#
# 에이전트에게 자연어로 질문하여 데이터를 분석합니다.

# %%
# 기본 통계 정보
query1 = "데이터의 기본 통계 정보를 보여주세요."
print(f"질문: {query1}\n")
result1 = agent.invoke({"messages": [HumanMessage(content=query1)]})
print(result1["messages"][-1].content)

# %%
# 특정 조건 필터링
query2 = "이익률이 30% 이상인 행을 찾아주세요."
print(f"\n질문: {query2}\n")
result2 = agent.invoke({"messages": [HumanMessage(content=query2)]})
print(result2["messages"][-1].content)

# %%
# 집계 분석
query3 = "제품별 평균 매출과 이익을 계산해주세요."
print(f"\n질문: {query3}\n")
result3 = agent.invoke({"messages": [HumanMessage(content=query3)]})
print(result3["messages"][-1].content)

# %%
# 복잡한 분석
query4 = "지역별로 매출이 가장 높은 날짜를 찾아주세요."
print(f"\n질문: {query4}\n")
result4 = agent.invoke({"messages": [HumanMessage(content=query4)]})
print(result4["messages"][-1].content)

# %% [markdown]
# ## 4. 여러 DataFrame 분석
#
# 여러 DataFrame을 동시에 분석할 수 있습니다.

# %%
# 두 번째 데이터셋 생성
data2 = {
    '제품': ['A', 'B', 'C'],
    '재고': [100, 150, 80],
    '가격': [50, 70, 90]
}
df2 = pd.DataFrame(data2)

# 여러 DataFrame을 전역 변수로 저장
_global_df2 = df2

# DataFrame 구조 정보 생성
df_info = f"""
첫 번째 DataFrame (df) 구조:
- 컬럼: {list(_global_df.columns)}
- 행 수: {len(_global_df)}
- 샘플 데이터:
{_global_df.head(3).to_string()}

두 번째 DataFrame (df2) 구조:
- 컬럼: {list(_global_df2.columns)}
- 행 수: {len(_global_df2)}
- 샘플 데이터:
{_global_df2.head(3).to_string()}
"""

# 여러 DataFrame을 사용하는 도구 생성
@tool
def pandas_query_multi(query: str) -> str:
    """여러 Pandas DataFrame에 대한 쿼리를 실행합니다.
    
    사용 가능한 DataFrame:
    - df: 첫 번째 DataFrame (매출 데이터)
      컬럼: {list(_global_df.columns)}
    - df2: 두 번째 DataFrame (재고 데이터)
      컬럼: {list(_global_df2.columns)}
    
    사용 예제:
    - df.head() - 첫 번째 DataFrame의 처음 5행 보기
    - df2.head() - 두 번째 DataFrame의 처음 5행 보기
    - df.groupby('제품')['매출'].sum() - 제품별 매출 합계
    - df2.groupby('제품')['재고'].sum() - 제품별 재고 합계
    - pd.merge(df.groupby('제품')['매출'].sum().reset_index(), 
               df2.groupby('제품')['재고'].sum().reset_index(), 
               on='제품', how='outer') - 두 DataFrame 병합
    
    Args:
        query: 실행할 pandas 코드 (df, df2, pd 변수 사용)
               반드시 유효한 Python 코드여야 합니다.
    
    Returns:
        쿼리 실행 결과 (문자열)
    """
    try:
        # 안전한 실행을 위한 제한된 네임스페이스
        safe_dict = {
            "df": _global_df,
            "df2": _global_df2,
            "pd": pd,
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "mean": lambda x: x.mean() if hasattr(x, 'mean') else None,
        }
        
        result = eval(query, {"__builtins__": {}}, safe_dict)
        
        if isinstance(result, pd.DataFrame):
            if len(result) > 20:
                return f"결과 (상위 20행):\n{result.head(20).to_string()}\n... (총 {len(result)}행)"
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except SyntaxError as e:
        return f"구문 오류: {str(e)}\n쿼리를 다시 확인해주세요. 예: df.head() 또는 df.groupby('제품')['매출'].sum()"
    except Exception as e:
        return f"오류 발생: {str(e)}\n사용 가능한 컬럼: df={list(_global_df.columns)}, df2={list(_global_df2.columns)}"

# 여러 DataFrame을 사용하는 에이전트
multi_agent = create_agent(
    model,
    tools=[pandas_query_multi],
    system_prompt=f"""당신은 데이터 분석 전문가입니다. 

{df_info}

두 개의 DataFrame을 분석할 때:
1. 먼저 각 DataFrame의 구조를 확인하세요 (df.head(), df2.head())
2. 공통 컬럼(제품)을 기준으로 데이터를 병합하세요
3. 제품별로 매출과 재고를 집계하세요
4. 결과를 명확하게 설명하세요

도구를 사용할 때는 정확한 Python 코드를 생성하세요."""
)

# 비교 분석
query7 = "첫 번째 데이터프레임과 두 번째 데이터프레임을 비교하여 제품별 매출과 재고를 분석해주세요."
print(f"질문: {query7}\n")
result7 = multi_agent.invoke({"messages": [HumanMessage(content=query7)]})
print(result7["messages"][-1].content)

# %% [markdown]
# ## 5. 실전 예제: CSV 파일 분석
#
# 실제 CSV 파일을 로드하여 분석합니다.

# %%
# CSV 파일 로드 (예제)
# 실제 파일이 있다면 주석을 해제하세요
# df_real = pd.read_csv("sales_data.csv")

# CSV 파일이 없는 경우 예제 데이터 사용
print("예제 데이터로 분석을 계속합니다...")

# 복잡한 분석 쿼리
query9 = """
다음 질문들에 답변해주세요:
1. 전체 기간의 총 매출과 총 이익은?
2. 평균 일일 매출은?
3. 이익률이 가장 높은 제품은?
4. 지역별 매출 분포는?
5. 향후 예측을 위한 제안사항은?
"""
print(f"질문: {query9}\n")

result9 = agent.invoke({"messages": [HumanMessage(content=query9)]})
print(result9["messages"][-1].content)

# %% [markdown]
# ## 6. 에이전트 제한사항 및 모범 사례
#
# ### 제한사항
# - 대용량 데이터셋에서는 성능 저하 가능
# - SQL 쿼리보다 느릴 수 있음
#
# ### 모범 사례
# - 데이터 전처리 후 에이전트에 전달
# - 명확하고 구체적인 질문 사용
# - 큰 데이터셋은 샘플링 후 분석
# - 중요한 분석은 결과 검증 필요

# %%
# 데이터 샘플링 예제
large_df = pd.DataFrame({
    '값': range(10000),
    '카테고리': ['A', 'B', 'C'] * 3333 + ['A']
})

# 샘플링
sample_df = large_df.sample(n=1000, random_state=42)

# 샘플 DataFrame을 전역 변수로 저장
_global_sample_df = sample_df

# 샘플 데이터용 도구 생성
@tool
def pandas_query_sample(query: str) -> str:
    """샘플 DataFrame에 대한 쿼리를 실행합니다.
    
    DataFrame 정보:
    - 컬럼: {list(_global_sample_df.columns)}
    - 행 수: {len(_global_sample_df)}
    
    사용 가능한 작업:
    - 데이터 조회: df.head(), df.tail(), df.describe()
    - 필터링: df[df['컬럼'] > 값]
    - 집계: df.groupby('컬럼').mean(), df.groupby('컬럼').sum()
    - 통계: df['컬럼'].mean(), df['컬럼'].max(), df['컬럼'].min()
    
    Args:
        query: 실행할 pandas 코드 (df 변수 사용)
               반드시 유효한 Python 코드여야 합니다.
    
    Returns:
        쿼리 실행 결과 (문자열)
    """
    try:
        # 안전한 실행을 위한 제한된 네임스페이스
        safe_dict = {
            "df": _global_sample_df,
            "pd": pd,
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "mean": lambda x: x.mean() if hasattr(x, 'mean') else None,
        }
        
        result = eval(query, {"__builtins__": {}}, safe_dict)
        
        if isinstance(result, pd.DataFrame):
            if len(result) > 20:
                return f"결과 (상위 20행):\n{result.head(20).to_string()}\n... (총 {len(result)}행)"
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except SyntaxError as e:
        return f"구문 오류: {str(e)}\n쿼리를 다시 확인해주세요. 예: df.head() 또는 df.groupby('카테고리')['값'].sum()"
    except Exception as e:
        return f"오류 발생: {str(e)}\n사용 가능한 컬럼: {list(_global_sample_df.columns)}"

# 샘플 데이터로 에이전트 생성
sample_df_info = f"""
샘플 DataFrame 구조:
- 컬럼: {list(_global_sample_df.columns)}
- 행 수: {len(_global_sample_df)}
- 샘플 데이터:
{_global_sample_df.head(3).to_string()}
"""

sample_agent = create_agent(
    model,
    tools=[pandas_query_sample],
    system_prompt=f"""당신은 데이터 분석 전문가입니다.

{sample_df_info}

pandas_query_sample 도구를 사용하여 분석하세요.
도구를 사용할 때는 정확한 Python 코드를 생성하세요."""
)

query10 = "데이터의 기본 통계를 보여주세요."
result10 = sample_agent.invoke({"messages": [HumanMessage(content=query10)]})
print(result10["messages"][-1].content)

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **Pandas DataFrame 에이전트**: 자연어로 데이터 분석 가능
# 2. **다중 DataFrame 지원**: 여러 데이터셋 비교 분석
# 3. **인사이트 도출**: 패턴 발견 및 제안사항 생성
# 4. **실전 활용**: CSV 파일 분석 및 리포트 생성
#
# **다음 단계**: 
# - [220_SQL_Agent.py](220_SQL_Agent.py)에서 SQL 데이터베이스 분석 학습
# - [streamlit-llm_LangChain/210_Data_Analysis_App.py](streamlit-llm_LangChain/210_Data_Analysis_App.py)에서 웹 UI 구현

# %%
