import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# 페이지 설정
st.set_page_config(
    page_title="기온 상승 트렌드 분석 웹앱",
    page_icon="🌡️",
    layout="wide"
)

# 제목 및 설명
st.title("🌡️ 1980년대 전후 기온 상승 속도 비교 분석")
st.markdown("""
제공된 대한민국 장기 기온 데이터를 바탕으로, **특정 시점(기본값: 1980년) 전후의 기온 상승 추세(기여도) 차이**를 검증하는 웹앱입니다.
가설대로 1980년 이후에 지구 온난화 및 도시화로 인해 기온 상승 속도가 빨라졌는지 확인해 보세요!
""")

# 데이터 로드 함수 (캐싱 적용으로 속도 향상)
@st.cache_data
def load_data():
    # 데이터 읽기 (제공된 파일 구조 반영)
    df = pd.read_csv("ta_20260601093156.csv", encoding="utf-8")
    
    # '날짜' 컬럼 앞뒤 공백 및 탭 문자 제거 제거
    df['날짜'] = df['날짜'].astype(str).str.strip()
    
    # datetime 변환 및 연도 추출
    df['Date'] = pd.to_datetime(df['날짜'], errors='coerce')
    df['연도'] = df['Date'].dt.year
    
    # 필요한 컬럼 정제 및 결측치 제거
    df = df.rename(columns={
        '평균기온(℃)': '평균기온',
        '최저기온(℃)': '최저기온',
        '최고기온(℃)': '최고기온'
    })
    
    # 연도별 평균 계산
    annual_data = df.groupby('연도').agg({
        '평균기온': 'mean',
        '최저기온': 'mean',
        '최고기온': 'mean'
    }).dropna().reset_index()
    
    return annual_data

# 데이터 로드 시도
try:
    data = load_data()
    
    # 사이드바 제어 요소
    st.sidebar.header("🛠️ 분석 설정")
    
    # 가설 기준 연도 선택 (기본값 1980년)
    min_year = int(data['연도'].min())
    max_year = int(data['연도'].max())
    split_year = st.sidebar.slider("가설 기준 연도 분할 지점", min_year + 10, max_year - 10, 1980)
    
    # 분석할 기온 지표 선택
    temp_target = st.sidebar.selectbox("분석할 기온 지표", ["평균기온", "최저기온", "최고기온"])

    # 데이터 분할
    before_df = data[data['연도'] < split_year]
    after_df = data[data['연도'] >= split_year]

    # 회귀 분석 수행 (선형 추세선 계산)
    slope_b, intercept_b, r_b, p_b, _ = linregress(before_df['연도'], before_df[temp_target])
    slope_a, intercept_a, r_a, p_a, _ = linregress(after_df['연도'], after_df[temp_target])

    # 대시보드 메트릭 카드 시각화
    st.subheader(f"📊 {split_year}년 전후 상승 속도 비교 요약")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{min_year}~{split_year-1}년 상승 속도",
            value=f"{slope_b*10:.3f} ℃ / 10년",
            delta=f"연간 {slope_b:.4f}℃"
        )
    with col2:
        # 상승 속도 변화율 계산
        change_rate = ((slope_a - slope_b) / abs(slope_b)) * 100 if slope_b != 0 else 0
        st.metric(
            label=f"{split_year}~{max_year}년 상승 속도",
            value=f"{slope_a*10:.3f} ℃ / 10년",
            delta=f"속도 증가율: {change_rate:+.1f}%",
            delta_color="inverse" if slope_a > slope_b else "normal"
        )
    with col3:
        avg_diff = after_df[temp_target].mean() - before_df[temp_target].mean()
        st.metric(
            label="두 구간의 평균 기온 차이",
            value=f"{avg_diff:+.2f} ℃",
            delta="이전 대비 이후 평균"
        )

    # 그래프 시각화 (Plotly 사용)
    st.subheader("📈 연도별 기온 변화 및 구간별 추세선")
    
    fig = go.Figure()

    # 실제 기온 데이터 산점도+선
    fig.add_trace(go.Scatter(
        x=data['연도'], y=data[temp_target],
        mode='markers+lines',
        name=f'연도별 {temp_target}',
        line=dict(color='lightgray', width=1.5),
        marker=dict(color='darkblue', size=5)
    ))

    # 기준 연도 이전 추세선
    x_b = before_df['연도']
    y_b_pred = slope_b * x_b + intercept_b
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b_pred,
        mode='lines',
        name=f'{split_year}년 이전 추세 (10년당 {slope_b*10:.2f}℃)',
        line=dict(color='blue', width=3, dash='dash')
    ))

    # 획기적인 기준 연도 이후 추세선
    x_a = after_df['연도']
    y_a_pred = slope_a * x_a + intercept_a
    fig.add_trace(go.Scatter(
        x=x_a, y=y_a_pred,
        mode='lines',
        name=f'{split_year}년 이후 추세 (10년당 {slope_a*10:.2f}℃)',
        line=dict(color='red', width=3)
    ))

    # 분할 기준선 분리 표시
    fig.add_vline(x=split_year, line_width=2, line_dash="dot", line_color="green", annotation_text=f"기준 연도 ({split_year})")

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title=f"{temp_target} (℃)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    # 결론 및 해석 데이터 제공
    st.subheader("🧐 데이터 분석 및 가설 검증 결과")
    st.markdown(f"""
    * **가설 검증:** 사용자가 지정한 **{split_year}년**을 기준으로 보았을 때, 이전의 기온 상승 속도는 10년당 **{slope_b*10:.3f}℃**였으나, 이후에는 10년당 **{slope_a*10:.3f}℃**로 변화했습니다.
    * **통계적 유의성(p-value):** * {split_year}년 이전 추세의 p-value: `{p_b:.5f}` {'(유의미함)' if p_b < 0.05 else '(통계적 유의성 부족)'}
        * {split_year}년 이후 추세의 p-value: `{p_a:.5f}` {'(유의미함)' if p_a < 0.05 else '(통계적 유의성 부족)'}
    """)
    
    # 데이터 테이블 보여주기
    with st.expander("📂 요약 데이터 테이블 보기"):
        st.dataframe(data.style.format({
            "평균기온": "{:.2f}℃",
            "최저기온": "{:.2f}℃",
            "최고기온": "{:.2f}℃"
        }), use_container_width=True)

except Exception as e:
    st.error(f"데이터를 읽거나 처리하는 중 오류가 발생했습니다. 파일명과 인코딩을 확인해 주세요. 오류 내용: {e}")
