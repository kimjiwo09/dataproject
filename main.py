import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. 페이지 설정
st.set_page_config(page_title="기온 상승 트렌드 분석기", layout="wide")

st.title("🌡️ 1980년대 전후 기온 상승 가속화 검증 앱")
st.markdown("""
제공된 기온 데이터를 바탕으로 **1980년 이전과 이후의 기온 상승 추세(기여도)**를 비교합니다.  
1980년대 이후 온난화 속도가 정말 빨라졌는지 추세선의 기울기로 확인해 보세요!
""")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    # CSV 파일 읽기
    df = pd.read_ta_20260601093156.csv(file_path)
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 컬럼의 앞뒤 공백 및 탭 문자 제거 후 datetime 변환
    df['날짜'] = df['날짜'].astype(str).str.replace(r'[\t\s]', '', regex=True)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    
    # 결측치 제거
    df = df.dropna(subset=['날짜', '평균기온(℃)'])
    
    # 연도 컬럼 추출
    df['연도'] = df['날짜'].dt.year
    
    # 연도별 평균 기온 계산
    annual_df = df.groupby('연도')['평균기온(℃)'].mean().reset_index()
    return annual_df

# 데이터 불러오기 (사용자가 업로드한 파일명 입력)
try:
    data_file = "ta_20260601093156.csv"
    df_yearly = load_data(data_file)
except Exception as e:
    st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 데이터 분할 (1980년 기준)
criterion_year = 1980
df_before = df_yearly[df_yearly['연도'] <= criterion_year]
df_after = df_yearly[df_yearly['연도'] >= criterion_year] # 1980년을 포함하여 연속성 유지

# 4. 회귀 분석 (추세선 계산 함수)
def get_trend_line(df):
    X = df['연도'].values.reshape(-1, 1)
    y = df['평균기온(℃)'].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    slope = model.coef_[0]
    return trend, slope

trend_before, slope_before = get_trend_line(df_before)
trend_after, slope_after = get_trend_line(df_after)

# 5. 대시보드 지표 (Metrics) 시각화
st.subheader("📊 10년당 기온 상승 폭 비교 (추세선 기울기)")
col1, col2, col3 = st.columns(3)

# 10년 단위 상승률로 변환 (기울기 * 10)
rate_before = slope_before * 10
rate_after = slope_after * 10
acceleration = rate_after / rate_before if rate_before != 0 else 0

with col1:
    st.metric(
        label=f"{df_yearly['연도'].min()}년 ~ {criterion_year}년 상승률", 
        value=f"{rate_before:.3f} ℃ / 10년"
    )
with col2:
    st.metric(
        label=f"{criterion_year}년 ~ {df_yearly['연도'].max()}년 상승률", 
        value=f"{rate_after:.3f} ℃ / 10년",
        delta=f"{rate_after - rate_before:.3f} ℃ 가속됨"
    )
with col3:
    st.metric(
        label="상승 속도 증가 배율", 
        value=f"{acceleration:.1f} 배 빨라짐"
    )

st.markdown("---")

# 6. Plotly를 이용한 대화형 그래프 시각화
st.subheader("📈 연도별 평균 기온 추세 및 1980년 전후 비교")

fig = go.Figure()

# 전체 연도 실제 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly['연도'], y=df_yearly['평균기온(℃)'],
    mode='markers', name='연평균 기온',
    marker=dict(color='gray', opacity=0.6)
))

# 1980년 이전 추세선
fig.add_trace(go.Scatter(
    x=df_before['연도'], y=trend_before,
    mode='lines', name=f'{criterion_year}년 이전 추세',
    line=dict(color='blue', width=3)
))

# 1980년 이후 추세선
fig.add_trace(go.Scatter(
    x=df_after['연도'], y=trend_after,
    mode='lines', name=f'{criterion_year}년 이후 추세',
    line=dict(color='red', width=3)
))

# 기준선(1980년) 수직선 표시
fig.add_vline(x=criterion_year, line_width=1.5, line_dash="dash", line_color="green")

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균 기온 (℃)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# 7. 가설 검증 결론 요약
st.subheader("💡 가설 검증 결론")
if slope_after > slope_before:
    st.success(f"""
    **가설이 성립합니다!** 데이터 분석 결과, 1980년 이후의 기온 상승 속도가 이전보다 **약 {acceleration:.1f}배** 더 가른 것으로 나타났습니다. 
    이는 전 세계적인 온난화 가속화 및 도시 열섬 현상의 심화 흐름과 일치합니다.
    """)
else:
    st.warning("1980년 이후 기온 상승률이 이전보다 가속되지 않았거나 유사한 흐름을 보입니다.")

# 데이터 내역 보여주기
with st.expander("📂 가공된 연도별 데이터 보기"):
    st.dataframe(df_yearly.set_index('연to').style.format("{:.2f}"))
