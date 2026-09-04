import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정
st.set_page_config(page_title="선제적 스마트 학교 재배치 의사결정 지원 시스템", layout="wide")

def calculate_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [float(lat1), float(lon1), np.array(lat2, dtype=float), np.array(lon2, dtype=float)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# 2. 데이터 로드 및 전처리 (지표 역전 및 군집 매핑 로직 개선)
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('학교정보_수정.csv')
    live_df = df[df['폐교여부'] == 0].copy()
    
    # 원본 정밀 위경도 확정
    live_df['위도'] = pd.to_numeric(live_df['위도'], errors='coerce')
    live_df['경도'] = pd.to_numeric(live_df['경도'], errors='coerce')
    
    # Min-Max Scaler 함수 (중앙값 결측치 보완 및 방향성 보정)
    def min_max(s, reverse=False):
        s_clean = s.fillna(s.median())
        min_v, max_v = s_clean.min(), s_clean.max()
        if max_v == min_v: 
            return s_clean * 0
        res = (s_clean - min_v) / (max_v - min_v) * 100.0
        return 100.0 - res if reverse else res

    # S1. 공간유휴화지수 (높을수록 위험)
    live_df['S1_유휴율'] = min_max(live_df['공간유휴화지수'].fillna(0))

    # S2. 전체학생수 (적을수록 취약하므로 reverse=True 적용)
    live_df['S2_학생수'] = min_max(live_df['전체학생수'].fillna(0), reverse=True)

    # S3. 교원1인당학생수 (소규모 소외학교 지원을 위해 적을수록 취약 -> reverse=True)
    live_df['S3_교원부담'] = min_max(live_df['교원1인당학생수'].fillna(0), reverse=True)

    # S4. 학급운영 취약성
    def get_s4(r):
        target = 6.0 if r['학교급'] == '초등학교' else 3.0
        actual = r['편성학급수']
        if pd.isna(actual): return 0.0
        return max(0.0, (target - actual) / target) * 100.0

    live_df['S4_학급취약'] = live_df.apply(get_s4, axis=1)

    # 종합 내부위험도 가중 합산
    live_df['내부위험도'] = (
        live_df['S1_유휴율'] * 0.25 +
        live_df['S2_학생수'] * 0.30 +
        live_df['S3_교원부담'] * 0.15 +
        live_df['S4_학급취약'] * 0.30
    ).fillna(0).round(1)

    live_df['대체학교수'] = live_df['동일생활권_대체학교수'].fillna(0)

    # K-Means 머신러닝 군집화 (표준화 스케일러 적용)
    X_2d = live_df[['내부위험도', '대체학교수']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2d)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    live_df['cluster'] = kmeans.fit_predict(X_scaled)

    # 각 클러스터 중심점의 상대적 위치(상대 평균값)를 기준으로 4대 정책유형 1:1 매핑
    centroids = live_df.groupby('cluster')[['내부위험도', '대체학교수']].mean()
    r_mean = centroids['내부위험도'].mean()
    a_mean = centroids['대체학교수'].mean()

    mapping = {}
    for c_id, row in centroids.iterrows():
        r, a = row['내부위험도'], row['대체학교수']
        if r >= r_mean and a >= a_mean:
            mapping[c_id] = '① 스마트 통합·재배치형'
        elif r >= r_mean and a < a_mean:
            mapping[c_id] = '② 필수 방어·지원형 (사각지대)'
        elif r < r_mean and a >= a_mean:
            mapping[c_id] = '③ 선제 관찰형'
        else:
            mapping[c_id] = '④ 지역 유지형'

    live_df['정책유형'] = live_df['cluster'].map(mapping)
    return df, live_df

raw_df, live_df = load_and_process_data()

st.sidebar.title("🎛️ 지역 및 학교 필터")
selected_sido = st.sidebar.multiselect("시·도 선택", options=sorted(live_df['시도'].dropna().unique()), default=[])
selected_grade = st.sidebar.multiselect("학교급 선택", options=['초등학교', '중학교', '고등학교'], default=['초등학교', '중학교', '고등학교'])

filtered_df = live_df.copy()
if selected_sido:
    filtered_df = filtered_df[filtered_df['시도'].isin(selected_sido)]
if selected_grade:
    filtered_df = filtered_df[filtered_df['학교급'].isin(selected_grade)]

st.title("🏫 선제적 지역 교육 인프라 보호 및 스마트 학교 재배치 시스템")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. 2D K-Means 클러스터링", 
    "🗺️ 2. 지도 기반 위험학교 분석", 
    "📋 3. 위험도(X축) 정렬 리스트", 
    "💡 4. 개별 학교 맞춤 진단 리포트"
])

ordered_types = [
    '① 스마트 통합·재배치형',
    '② 필수 방어·지원형 (사각지대)',
    '③ 선제 관찰형',
    '④ 지역 유지형'
]

# Tab 1
with tab1:
    st.subheader("📌 [X축: 신규 종합 위험도 vs Y축: 동일생활권 대체학교수] 머신러닝 군집화")
    st.caption("공식: S1(공간유휴율 25%) + S2(학생수결손 30%) + S3(교원부담 15%) + S4(학급취약 30%)")
    
    fig_scatter = px.scatter(
        filtered_df, 
        x='내부위험도', 
        y='대체학교수', 
        color='정책유형',
        category_orders={'정책유형': ordered_types},
        hover_name='학교명',
        hover_data=['시도', '행정구', '전체학생수', '공간유휴화지수', '학교코드_KEDI'],
        color_discrete_map={
            '① 스마트 통합·재배치형': '#d9534f',
            '② 필수 방어·지원형 (사각지대)': '#f0ad4e',
            '③ 선제 관찰형': '#428bca',
            '④ 지역 유지형': '#5cb85c'
        },
        labels={'내부위험도': '내부 종합 위험도 점수 (X축)', '대체학교수': '동일생활권 대체학교 수 (Y축)'},
        height=600
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 정책 유형별 학교 분포 현황")
    
    counts_map = filtered_df['정책유형'].value_counts().to_dict()
    total_live = len(filtered_df)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("① 스마트 통합·재배치형", f"{counts_map.get(ordered_types[0], 0):,}교", f"전국 비중 {counts_map.get(ordered_types[0], 0)/total_live*100:.1f}%", delta_color="off")
    m2.metric("② 필수 방어·지원형", f"{counts_map.get(ordered_types[1], 0):,}교", f"전국 비중 {counts_map.get(ordered_types[1], 0)/total_live*100:.1f}%", delta_color="off")
    m3.metric("③ 선제 관찰형", f"{counts_map.get(ordered_types[2], 0):,}교", f"전국 비중 {counts_map.get(ordered_types[2], 0)/total_live*100:.1f}%", delta_color="off")
    m4.metric("④ 지역 유지형", f"{counts_map.get(ordered_types[3], 0):,}교", f"전국 비중 {counts_map.get(ordered_types[3], 0)/total_live*100:.1f}%", delta_color="off")

# Tab 2
with tab2:
    st.subheader("🗺️ 대한민국 실제 지도 기반 한계학교 핑(Marker) 분석")
    st.caption("마우스 휠이나 지도 조작 버튼을 통해 확대·축소가 가능하며, 핑 호버/클릭 시 상세 정보가 표출됩니다.")
    
    map_type_filter = st.multiselect(
        "지도에 표시할 정책 유형 선택:",
        options=ordered_types,
        default=['① 스마트 통합·재배치형']
    )
    
    map_df = filtered_df[filtered_df['정책유형'].isin(map_type_filter)].copy()
    
    fig_real_map = px.scatter_mapbox(
        map_df,
        lat="위도",
        lon="경도",
        color="정책유형",
        category_orders={'정책유형': ordered_types},
        size="전체학생수",
        hover_name="학교명",
        hover_data={
            "학교코드_KEDI": True,
            "시도": True,
            "행정구": True,
            "학교급": True,
            "전체학생수": True,
            "내부위험도": True,
            "대체학교수": True,
            "위도": False,
            "경도": False
        },
        color_discrete_map={
            '① 스마트 통합·재배치형': '#d9534f',
            '② 필수 방어·지원형 (사각지대)': '#f0ad4e',
            '③ 선제 관찰형': '#428bca',
            '④ 지역 유지형': '#5cb85c'
        },
        zoom=6.5,
        center={"lat": 36.5, "lon": 127.5},
        mapbox_style="open-street-map",
        height=700,
        title="전국 한계학교 지리적 위치 및 정책 유형 마커 지도"
    )
    
    fig_real_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_real_map, use_container_width=True, config={'scrollZoom': True})

# Tab 3
with tab3:
    st.subheader("📋 내부 위험도(X축) 내림차순 학교 리스트")
    
    col_sort, _ = st.columns([1, 2])
    with col_sort:
        sort_col = st.selectbox("정렬 기준 선택", ["내부위험도 (위험도 높은 순)", "전체학생수 (학생수 적은 순)", "공간유휴화지수 (유휴율 높은 순)"])
    
    if "내부위험도" in sort_col:
        sorted_list = filtered_df.sort_values(by='내부위험도', ascending=False)
    elif "전체학생수" in sort_col:
        sorted_list = filtered_df.sort_values(by='전체학생수', ascending=True)
    else:
        sorted_list = filtered_df.sort_values(by='공간유휴화지수', ascending=False)
        
    display_cols = ['학교코드_KEDI', '학교명', '시도', '행정구', '학교급', '내부위험도', '전체학생수', '전체교원수', '편성학급수', '총교실수', '공간유휴화지수', '대체학교수', '정책유형']
    
    st.dataframe(
        sorted_list[display_cols],
        column_config={
            "학교코드_KEDI": st.column_config.TextColumn("KEDI 학교코드"),
            "내부위험도": st.column_config.ProgressColumn("위험도 점수", format="%f점", min_value=0, max_value=100),
            "공간유휴화지수": st.column_config.NumberColumn("공간유휴율", format="%.2f"),
            "전체학생수": st.column_config.NumberColumn("학생수(명)"),
            "전체교원수": st.column_config.NumberColumn("교원수(명)"),
            "편성학급수": st.column_config.NumberColumn("학급수(개)"),
            "총교실수": st.column_config.NumberColumn("총교실수(개)"),
            "대체학교수": st.column_config.NumberColumn("동일생활권 대체교(개)"),
        },
        height=500,
        use_container_width=True
    )

# Tab 4
with tab4:
    st.subheader("🔍 개별 학교 정밀 진단 리포트")
    selected_school = st.selectbox("진단할 학교를 선택하세요:", options=sorted(filtered_df['학교명'].unique()))
    
    if selected_school:
        s_info = filtered_df[filtered_df['학교명'] == selected_school].iloc[0]
        
        c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1.2, 1.2, 1])
        c1.metric("KEDI 학교코드 / 위치", f"{s_info['학교코드_KEDI']} ({s_info['시도']} {s_info['행정구']})")
        c2.metric("내부 종합 위험도", f"{s_info['내부위험도']}점")
        c3.metric("전교생 / 전체교원수", f"{int(s_info['전체학생수'])}명 / {int(s_info['전체교원수'])}명")
        c4.metric("학급수 / 총교실수", f"{int(s_info['편성학급수'])}학급 / {int(s_info['총교실수'])}실")
        c5.metric("동일생활권 대체교", f"{int(s_info['대체학교수'])}개교")
        
        st.markdown("---")
        
        st.write("### 📌 4대 세부 위험도 지표 점수")
        radar_df = pd.DataFrame({
            '지표': ['S1. 공간유휴화율', 'S2. 학생수결손율', 'S3. 교원자원부담', 'S4. 학급운영취약성'],
            '점수': [s_info['S1_유휴율'], s_info['S2_학생수'], s_info['S3_교원부담'], s_info['S4_학급취약']]
        })
        fig_bar = px.bar(radar_df, x='지표', y='점수', color='지표', text_auto='.1f', title=f"{selected_school} ({s_info['학교코드_KEDI']}) 세부 위험 지표 점수")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🏫 통폐합/재배치 시 학생 수용 대안: 근처 동일 학교급 추천 TOP 5")
        st.caption(f"💡 **대체 학교 마커에 마우스를 올려보세요!** **{selected_school}**과의 정밀 직선거리가 팝업 형태로 표출됩니다.")
        
        same_grade_df = live_df[
            (live_df['학교급'] == s_info['학교급']) & 
            (live_df['학교명'] != selected_school) &
            (live_df['위도'].notna()) & 
            (live_df['경도'].notna())
        ].copy()
        
        same_grade_df['직선거리_km'] = calculate_haversine(
            s_info['위도'], s_info['경도'],
            same_grade_df['위도'].values, same_grade_df['경도'].values
        ).round(2)
        
        top5_recommend = same_grade_df.sort_values('직선거리_km').head(5).copy()
        
        all_lats = [s_info['위도']] + top5_recommend['위도'].tolist()
        all_lons = [s_info['경도']] + top5_recommend['경도'].tolist()
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        max_diff = max(max(all_lats) - min(all_lats), max(all_lons) - min(all_lons))
        auto_zoom = 11.5 if max_diff < 0.1 else (10.2 if max_diff < 0.3 else 8.8)

        fig_rec_map = go.Figure()
        
        # 1. 진단 대상 학교 (붉은색 마커)
        fig_rec_map.add_trace(go.Scattermapbox(
            lat=[s_info['위도']],
            lon=[s_info['경도']],
            mode='markers',
            marker=dict(size=26, color='#FF0000', opacity=1.0),
            name=f"🎯 [진단대상] {selected_school}",
            hoverinfo="text",
            hovertext=f"🎯 <b>[진단대상] {selected_school} ({s_info['학교코드_KEDI']})</b><br>위치: {s_info['시도']} {s_info['행정구']}<br>학생수: {int(s_info['전체학생수'])}명 | 교원수: {int(s_info['전체교원수'])}명<br>내부위험도: {s_info['내부위험도']}점"
        ))
        
        # 2. 추천 대체 학교 5곳
        for idx, (_, r) in enumerate(top5_recommend.iterrows(), 1):
            fig_rec_map.add_trace(go.Scattermapbox(
                lat=[r['위도']],
                lon=[r['경도']],
                mode='markers',
                marker=dict(size=22, color='#0066FF', opacity=0.9),
                name=f"{idx}. {r['학교명']}",
                hoverinfo="text",
                hovertext=(
                    f"📍 <b>{idx}. {r['학교명']} ({r['학교코드_KEDI']})</b><br>"
                    f"──────────────────────<br>"
                    f"🔗 <b>진단대상 [{selected_school}]과의 직선거리: {r['직선거리_km']} km</b><br>"
                    f"👥 학생수: {int(r['전체학생수'])}명 | 교원수: {int(r['전체교원수'])}명 | 총교실수: {int(r['총교실수'])}실<br>"
                    f"⚠️ 위험도: {r['내부위험도']}점"
                )
            ))
        
        fig_rec_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=auto_zoom,
                center={"lat": center_lat, "lon": center_lon}
            ),
            margin={"r":0, "t":10, "l":0, "b":0},
            height=480,
            legend=dict(
                title="<b>📍 표시 학교 목록</b>",
                yanchor="top", y=0.98,
                xanchor="left", x=0.02,
                bgcolor="rgba(255, 255, 255, 0.9)"
            )
        )
        
        st.plotly_chart(fig_rec_map, use_container_width=True, config={'scrollZoom': True})
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"### 📋 **{selected_school}** 인근 추천 대체학교 TOP 5 상세 명단")
        
        rec_cols = ['학교코드_KEDI', '학교명', '직선거리_km', '전체학생수', '전체교원수', '총교실수', '내부위험도']
        st.dataframe(
            top5_recommend[rec_cols],
            column_config={
                "학교코드_KEDI": st.column_config.TextColumn("KEDI 학교코드"),
                "학교명": st.column_config.TextColumn("학교명"),
                "직선거리_km": st.column_config.NumberColumn("진단대상과의 직선거리 (km)", format="%.2f km"),
                "전체학생수": st.column_config.NumberColumn("전체 학생수 (명)"),
                "전체교원수": st.column_config.NumberColumn("전체 교원수 (명)"),
                "총교실수": st.column_config.NumberColumn("총교실수 (실)"),
                "내부위험도": st.column_config.ProgressColumn("내부 종합 위험도", format="%f점", min_value=0, max_value=100)
            },
            use_container_width=True,
            height=215
        )