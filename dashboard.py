import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import time
import feedparser
import requests
import random
from datetime import datetime

# TrendOps 모듈 임포트
from trendops.collector.collector_rss import RSSCollector
from trendops.analyst.structured_analyzer import StructuredAnalyzer
from trendops.store.vector_store import get_vector_store

#streamlit run dashboard.py

# -----------------------------------------------------------------------------
# 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrendOps AI Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 세션 상태 초기화
# -----------------------------------------------------------------------------
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'articles_data' not in st.session_state:
    st.session_state.articles_data = None
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
if 'current_keyword' not in st.session_state:
    st.session_state.current_keyword = ""

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def run_async(coro):
    """비동기 실행 래퍼"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@st.cache_data(ttl=600)
def get_realtime_trends(geo="KR"):
    """
    [핵심 수정] trigger_google.py의 성공 URL 적용
    URL: https://trends.google.com/trending/rss?geo=KR
    """
    try:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # ✅ trigger_google.py에서 확인된 '진짜' 작동하는 주소
        url = f"https://trends.google.com/trending/rss?geo={geo}"
        
        # ✅ trigger_google.py의 헤더 설정 그대로 적용
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        
        # 1. Requests로 데이터 요청
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            # 200 OK가 아니면 에러 메시지 반환
            return [], f"HTTP Error: {response.status_code}"

        # 2. 파싱
        feed = feedparser.parse(response.content)
        
        if not feed.entries:
            return [], "Empty Feed (데이터 없음)"
            
        trends = [entry.title for entry in feed.entries]
        return trends, current_time
        
    except Exception as e:
        return [], f"Error: {str(e)}"

# -----------------------------------------------------------------------------
# 사이드바
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔥 TrendOps Control")
    st.markdown("---")
    
    st.subheader("📈 Real-time Trends")
    
    # 새로고침 버튼
    col_ref, col_time = st.columns([1, 1])
    with col_ref:
        if st.button("🔄 새로고침", use_container_width=True):
            get_realtime_trends.clear()
            st.rerun()
            
    trends, status_msg = get_realtime_trends()
    
    with col_time:
        if "Error" in status_msg or "HTTP" in status_msg:
             st.error("Error")
        else:
             st.caption(f"Update:\n{status_msg}")

    st.markdown("👇 **분석할 키워드 클릭**")
    
    # 에러 메시지가 있으면 자세히 표시
    if "Error" in status_msg:
        st.error(status_msg)
    elif not trends:
        st.warning("데이터 로딩 중...")
    
    # 트렌드 목록 출력
    for keyword in trends:
        if st.button(f"🔥 {keyword}", key=f"trend_{keyword}", use_container_width=True):
            st.session_state.current_keyword = keyword
            st.session_state.analysis_result = None
            st.session_state.image_result = None
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    model_name = st.selectbox("LLM Model", ["exaone3.5"], index=0)
    max_articles = st.slider("Max Articles", 5, 50, 20)
    enable_image = st.checkbox("Card News", value=True)

# -----------------------------------------------------------------------------
# 메인 화면
# -----------------------------------------------------------------------------
st.title("📊 TrendOps: 지능형 트렌드 분석")

tab1, tab2, tab3 = st.tabs(["🚀 분석 실행", "🗄️ DB 확인", "🛠️ 시스템 상태"])

# TAB 1: 분석
with tab1:
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        keyword = st.text_input(
            "키워드 입력", 
            value=st.session_state.current_keyword,
            placeholder="사이드바에서 트렌드를 선택하거나 직접 입력하세요",
            label_visibility="collapsed"
        )
        
    with col_btn:
        analyze_btn = st.button("🚀 분석 시작", use_container_width=True, type="primary")

    if analyze_btn and keyword:
        st.session_state.current_keyword = keyword
        status_container = st.container()
        
        with status_container:
            with st.status("🕵️ 에이전트가 작업 중입니다...", expanded=True) as status:
                try:
                    # 1. 수집
                    st.write(f"🔍 Google News 수집 중: '{keyword}'")
                    async def fetch():
                        async with RSSCollector(max_results=max_articles) as c:
                            return await c.fetch(keyword)
                    
                    documents = run_async(fetch())
                    
                    if not documents:
                        status.update(label="❌ 기사 없음", state="error")
                        st.error("관련 기사를 찾을 수 없습니다.")
                        st.stop()
                    
                    st.write(f"✅ {len(documents)}건 수집 완료")
                    
                    articles_list = [
                        {
                            "title": d.title, 
                            "summary": d.summary, 
                            "source": d.source, 
                            "published": str(d.published),
                            # [중요] 중복 제거용 keyword 추가
                            "keyword": keyword 
                        }
                        for d in documents
                    ]
                    st.session_state.articles_data = articles_list

                    # 2. 분석
                    st.write(f"🧠 {model_name} 분석 중...")
                    async def analyze():
                        async with StructuredAnalyzer(model_name=model_name) as a:
                            return await a.analyze(keyword, articles_list)
                    
                    an_res = run_async(analyze())
                    st.session_state.analysis_result = an_res
                    st.write("✅ 분석 완료")

                    status.update(label="🎉 작업 완료!", state="complete")
                    
                except Exception as e:
                    status.update(label="⚠️ 에러 발생", state="error")
                    st.error(f"Error: {e}")
                    st.stop()

    # 결과 표시
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result.analysis
        
        st.divider()
        st.header(f"🔥 분석 리포트: {st.session_state.current_keyword}")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        
        with c1:
            st.markdown("### 📌 핵심 원인")
            st.info(res.main_cause)
            st.markdown("### 📝 3줄 요약")
            st.write(res.summary)

        with c2:
            st.markdown("### 📊 감성 분석")
            sent_data = res.sentiment_ratio.model_dump()
            fig = px.pie(
                values=list(sent_data.values()), 
                names=list(sent_data.keys()),
                color=list(sent_data.keys()),
                color_discrete_map={'positive':'#4ade80', 'negative':'#f87171', 'neutral':'#9ca3af'},
                hole=0.6
            )
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
            st.plotly_chart(fig, use_container_width=True)


        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### 💬 주요 반응")
            for op in res.key_opinions:
                st.success(f"• {op}")
        
        with c4:
            st.markdown("### 📰 원본 뉴스")
            if st.session_state.articles_data:
                df = pd.DataFrame(st.session_state.articles_data)
                st.dataframe(df[['title', 'source']], height=200, hide_index=True)

# TAB 2: DB
with tab2:
    st.subheader("🗄️ 벡터 DB 상태")
    try:
        store = get_vector_store()
        stats = store.get_stats()
        st.metric("총 문서 수", stats.count)
        st.caption(f"Path: {store.persist_path}")
    except:
        st.warning("DB 없음")

# TAB 3: 시스템
with tab3:
    st.subheader("🖥️ 시스템 상태")
    try:
        import requests
        res = requests.get("http://localhost:11434/api/tags")
        if res.status_code == 200:
            st.success("✅ Ollama 연결됨")
            st.json([m['name'] for m in res.json()['models']])
    except:
        st.error("❌ Ollama 연결 실패")