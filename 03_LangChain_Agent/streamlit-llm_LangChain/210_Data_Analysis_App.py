#---------------------------------------------------------
# Streamlit 데이터 분석 에이전트 앱
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="데이터 분석 에이전트", page_icon=":chart_with_upwards_trend:")
st.markdown("<h1 style='text-align: center;'>데이터 분석 에이전트</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-nano", model_provider="openai")

if "agent" not in st.session_state:
    st.session_state.agent = None

if "df" not in st.session_state:
    st.session_state.df = None

# ---------------------------------------------------------------------------------
# Pandas 도구 생성 함수
# ---------------------------------------------------------------------------------
def create_pandas_tool(df):
    """DataFrame을 위한 pandas 도구를 생성합니다."""
    # DataFrame을 클로저로 캡처
    _df = df.copy()
    
    @tool
    def pandas_query(query: str) -> str:
        """Pandas DataFrame에 대한 쿼리를 실행합니다.
        
        이 도구는 pandas DataFrame을 조작하고 분석하는 데 사용됩니다.
        
        DataFrame 정보:
        - 컬럼: {list(_df.columns)}
        - 행 수: {len(_df)}
        
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
        # 안전한 네임스페이스에서 pandas 쿼리 실행
        try:
            # 안전한 실행을 위한 제한된 네임스페이스
            safe_dict = {
                "df": _df,
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
            return f"오류 발생: {str(e)}\n사용 가능한 컬럼: {list(_df.columns)}"
    
    return pandas_query

def create_dataframe_agent(llm, df):
    """DataFrame을 위한 에이전트를 생성합니다."""
    # pandas 도구 생성
    pandas_tool = create_pandas_tool(df)
    
    # DataFrame 정보 생성
    df_info = f"""
DataFrame 구조:
- 컬럼: {list(df.columns)}
- 행 수: {len(df)}
- 샘플 데이터:
{df.head(3).to_string()}
"""
    
    # 시스템 프롬프트
    system_prompt = f"""당신은 데이터 분석 전문가입니다.

{df_info}

사용자의 질문에 답하기 위해 pandas_query 도구를 사용하여 DataFrame을 분석하세요.
결과를 명확하고 이해하기 쉽게 설명하세요.
도구를 사용할 때는 정확한 Python 코드를 생성하세요."""
    
    # 에이전트 생성
    agent = create_agent(
        llm,
        tools=[pandas_tool],
        system_prompt=system_prompt
    )
    
    return agent

# ---------------------------------------------------------------------------------
# 사이드바: 데이터 업로드
# ---------------------------------------------------------------------------------
st.sidebar.title("📊 데이터 관리")

uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    # CSV 파일 읽기 및 에이전트 재생성
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        st.session_state.df = df
        st.session_state.agent = create_dataframe_agent(
            st.session_state.llm,
            df
        )
        st.sidebar.success(f"✅ {len(df)}행, {len(df.columns)}열 로드 완료")
    except Exception as e:
        st.sidebar.error(f"오류: {str(e)}")

# 예제 데이터 로드
if st.sidebar.button("예제 데이터 로드"):
    data = {
        '날짜': pd.date_range('2024-01-01', periods=30, freq='D'),
        '매출': [1000 + i*50 + (i%7)*100 for i in range(30)],
        '비용': [500 + i*20 + (i%5)*50 for i in range(30)],
        '제품': ['A', 'B', 'C'] * 10,
        '지역': ['서울', '부산', '대구'] * 10
    }
    df = pd.DataFrame(data)
    # 파생 컬럼 계산 (이익, 이익률)
    df['이익'] = df['매출'] - df['비용']
    df['이익률'] = (df['이익'] / df['매출'] * 100).round(2)
    
    st.session_state.df = df
    st.session_state.agent = create_dataframe_agent(
        st.session_state.llm,
        df
    )
    st.sidebar.success("✅ 예제 데이터 로드 완료")

# 데이터 미리보기
if st.session_state.df is not None:
    st.sidebar.subheader("데이터 미리보기")
    st.sidebar.dataframe(st.session_state.df.head(), use_container_width=True)

# ---------------------------------------------------------------------------------
# 메인 영역: 분석
# ---------------------------------------------------------------------------------
if st.session_state.df is None:
    st.info("👈 사이드바에서 CSV 파일을 업로드하거나 예제 데이터를 로드하세요.")
else:
    st.subheader("데이터 분석")
    
    # 데이터 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("행 수", len(st.session_state.df))
    with col2:
        st.metric("열 수", len(st.session_state.df.columns))
    with col3:
        st.metric("데이터 타입", st.session_state.df.dtypes.value_counts().sum())
    
    # 분석 질문 입력
    query = st.text_area(
        "분석할 내용을 자연어로 입력하세요:",
        placeholder="예: 제품별 평균 매출을 계산해주세요.",
        height=100
    )
    
    if st.button("🔍 분석 실행", type="primary") and query:
        with st.spinner("분석 중..."):
            # 에이전트 실행 및 결과 추출
            try:
                result = st.session_state.agent.invoke({
                    "messages": [HumanMessage(content=query)]
                })
                st.success("분석 완료!")
                # 에이전트 응답에서 마지막 메시지의 내용 추출
                if "messages" in result and len(result["messages"]) > 0:
                    response_content = result["messages"][-1].content
                    st.write(response_content)
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"분석 오류: {str(e)}")
    
    # 데이터 표시
    with st.expander("전체 데이터 보기"):
        st.dataframe(st.session_state.df, use_container_width=True)
