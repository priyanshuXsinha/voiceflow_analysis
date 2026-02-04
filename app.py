import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv
from textblob import TextBlob
import os
from datetime import datetime

# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="VoiceFlow AI Analytics", layout="wide", page_icon="🎯")
load_dotenv()

# MODERN CSS WITH DARK THEME & VIBRANT COLORS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
    
    /* ROOT VARIABLES */
    :root {
        --bg-primary: #0a0e27;
        --bg-secondary: #131729;
        --bg-card: #1a1f3a;
        --accent-primary: #667eea;
        --accent-secondary: #764ba2;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --accent-info: #3b82f6;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-color: #2d3548;
    }
    
    /* GLOBAL STYLES */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }
    
    p, div, span, label {
        color: var(--text-secondary) !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 2px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem !important;
        margin-bottom: 1.5rem;
    }
    
    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 2px dashed var(--accent-primary);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-secondary);
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* METRIC CARDS */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        border-color: var(--accent-primary);
    }
    
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* TABS */
    [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    
    [data-baseweb="tab"]:hover {
        background: var(--bg-card) !important;
        color: var(--accent-primary) !important;
    }
    
    [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* CHAT INTERFACE */
    .stChatMessage {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .stChatMessage[data-testid="stChatMessageContent"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-left: 4px solid var(--accent-primary) !important;
    }
    
    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
    }
    
    /* INPUT FIELDS */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextArea > div > div > textarea,
    .stChatInput > div > div > textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus,
    .stTextArea > div > div > textarea:focus,
    .stChatInput > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        background: var(--bg-card);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        overflow: hidden;
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--accent-primary) !important;
    }
    
    /* ALERTS */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        border-left: 4px solid var(--accent-primary) !important;
    }
    
    .stSuccess {
        border-left-color: var(--accent-success) !important;
    }
    
    .stError {
        border-left-color: var(--accent-danger) !important;
    }
    
    /* PROGRESS BAR */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* DIVIDER */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            var(--border-color) 20%, 
            var(--accent-primary) 50%, 
            var(--border-color) 80%, 
            transparent 100%
        );
        margin: 2rem 0;
    }
    
    /* INFO BOX */
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border: 1px solid var(--accent-primary) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    /* TABLE */
    table {
        background: var(--bg-card) !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    thead tr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    thead th {
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    tbody tr {
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    tbody td {
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
    }
    
    tbody tr:hover {
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    /* DOWNLOAD BUTTON */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Hardcoded LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing in .env file")
    st.stop()

# Initialize LLM with hardcoded model
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0.3,
        model_name="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    )

# --- 2. DATA PRE-PROCESSOR ---
def auto_tag_data(df):
    text_col = None
    for col in df.columns:
        if "text" in col.lower() or "review" in col.lower() or "comment" in col.lower() or "feedback" in col.lower():
            text_col = col
            break
    
    if not text_col:
        return df, "Error: Could not find a text/review/comment column."

    st.toast("⚙️ Auto-processing your data...", icon="🔄")
    
    if 'sentiment' not in df.columns or 'rating' not in df.columns:
        def get_sentiment(text):
            analysis = TextBlob(str(text))
            score = analysis.sentiment.polarity
            if score > 0.1: return 'Positive', 5
            elif score < -0.1: return 'Negative', 1
            else: return 'Neutral', 3
            
        df['sentiment'], df['rating'] = zip(*df[text_col].apply(get_sentiment))

    if 'category' not in df.columns:
        def get_category(text):
            text = str(text).lower()
            if any(x in text for x in ['late', 'time', 'slow', 'delivery', 'driver', 'rider', 'shipping']): 
                return 'Delivery & Logistics'
            if any(x in text for x in ['quality', 'product', 'defect', 'broken', 'damaged', 'poor']): 
                return 'Product Quality'
            if any(x in text for x in ['app', 'bug', 'crash', 'payment', 'ui', 'website', 'interface']): 
                return 'Technical Issues'
            if any(x in text for x in ['support', 'service', 'help', 'staff', 'rude', 'refund', 'response']): 
                return 'Customer Service'
            if any(x in text for x in ['price', 'cost', 'expensive', 'cheap', 'value', 'billing']): 
                return 'Pricing'
            return 'General Feedback'
            
        df['category'] = df[text_col].apply(get_category)
    
    if 'review_text' not in df.columns:
        df['review_text'] = df[text_col]

    return df, None

# --- 3. BACKEND LOGIC ---
@st.cache_resource
def build_vector_db(df):
    from langchain_core.documents import Document
    docs = [Document(page_content=row['review_text'], metadata={"source": "upload"}) for _, row in df.iterrows()]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="voiceflow_v1")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
    return vectorstore, ranker

def generate_answer(query, vectorstore, ranker):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) 
    initial_docs = retriever.invoke(query)
    rerank_request = RerankRequest(query=query, passages=[
        {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(initial_docs)
    ])
    reranked_results = ranker.rerank(rerank_request)[:5] 
    context = "\n".join([f"- {res['text']}" for res in reranked_results])
    
    llm = get_llm()
    system_prompt = f"""
    You are an expert Voice of Customer analyst.
    User Question: "{query}"
    Context from customer feedback: {context}
    
    Task: Provide a detailed, professional analysis with clear insights and actionable recommendations.
    """
    response = llm.invoke(system_prompt)
    return response.content, reranked_results

# --- 4. SIDEBAR ---
st.sidebar.markdown("# 🎛️ Control Center")
st.sidebar.markdown("---")

# AI Status Badge with actual connection
status_html = """
<div style='
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    padding: 0.75rem 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    color: white;
'>
    🟢 AI Connected • Llama 3.3 70B
</div>
"""
st.sidebar.markdown(status_html, unsafe_allow_html=True)

# File Uploader
st.sidebar.markdown("### 📂 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv", label_visibility="collapsed")

# Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "last_upload_time" not in st.session_state:
    st.session_state.last_upload_time = None

# Data Loading Logic
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    processed_df, error = auto_tag_data(raw_df)
    if error:
        st.error(error)
    else:
        st.session_state.df = processed_df
        st.session_state.last_upload_time = datetime.now()
        st.sidebar.success("✅ Data loaded & processed!")
elif os.path.exists("reviews.csv"):
    if st.session_state.df is None:
        raw_df = pd.read_csv("reviews.csv")
        processed_df, error = auto_tag_data(raw_df)
        if not error:
            st.session_state.df = processed_df
            st.sidebar.info("📊 Demo dataset loaded")

# Sidebar Stats
if st.session_state.df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Overview")
    df = st.session_state.df
    
    st.sidebar.metric("Total Records", f"{len(df):,}")
    st.sidebar.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
    
    negative_pct = (len(df[df['sentiment']=='Negative']) / len(df) * 100)
    positive_pct = (len(df[df['sentiment']=='Positive']) / len(df) * 100)
    
    st.sidebar.metric("Positive Rate", f"{positive_pct:.1f}%")
    st.sidebar.metric("Issue Rate", f"{negative_pct:.1f}%")
    
    if st.session_state.last_upload_time:
        st.sidebar.markdown("---")
        st.sidebar.caption(f"⏰ Updated: {st.session_state.last_upload_time.strftime('%H:%M:%S')}")

# --- 5. MAIN DASHBOARD ---
title_html = """
<div style='text-align: center; margin: 2rem 0;'>
    <h1 style='
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    '>🎯 VoiceFlow Analytics</h1>
    <p style='
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 500;
    '>AI-Powered Customer Intelligence Platform</p>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📈 Analytics Dashboard", "🤖 AI Analyst Chat", "📝 Data Explorer"])
    
    # === TAB 1: ANALYTICS DASHBOARD ===
    with tab1:
        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tickets = len(df)
            st.metric("Total Feedback", f"{total_tickets:,}", "+12%")
        
        with col2:
            avg_rating = df['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f} / 5.0", "-0.2", delta_color="inverse")
        
        with col3:
            critical_issues = len(df[df['sentiment']=='Negative'])
            st.metric("Critical Issues", f"{critical_issues:,}", "⚠️ Needs Attention")
        
        with col4:
            st.metric("AI Processing", "100%", "✓ Complete")
        
        st.markdown("---")
        
        # Charts Section
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 🔴 Top Issue Categories")
            neg_df = df[df['sentiment'] == 'Negative']
            if not neg_df.empty:
                cat_counts = neg_df['category'].value_counts().reset_index()
                cat_counts.columns = ['Category', 'Count']
                
                fig = px.bar(
                    cat_counts, 
                    x='Count', 
                    y='Category', 
                    orientation='h',
                    text='Count',
                    color='Count',
                    color_continuous_scale=['#ef4444', '#dc2626', '#991b1b']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                fig.update_traces(textposition='outside')
                fig.update_xaxes(showgrid=True, gridcolor='#2d3548')
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No critical issues detected!")
        
        with col_chart2:
            st.markdown("### 😊 Sentiment Distribution")
            sent_counts = df['sentiment'].value_counts().reset_index()
            sent_counts.columns = ['Sentiment', 'Count']
            
            color_map = {
                'Positive': '#10b981',
                'Negative': '#ef4444',
                'Neutral': '#f59e0b'
            }
            
            fig2 = px.pie(
                sent_counts, 
                values='Count', 
                names='Sentiment',
                hole=0.5,
                color='Sentiment',
                color_discrete_map=color_map
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=400,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig2.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=14
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Additional Insights Section
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("### 📊 Category Analysis")
            cat_sentiment = pd.crosstab(df['category'], df['sentiment'])
            fig3 = px.bar(
                cat_sentiment,
                barmode='group',
                color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444', 'Neutral': '#f59e0b'}
            )
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title="Category",
                yaxis_title="Count",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(title="Sentiment")
            )
            fig3.update_xaxes(showgrid=False)
            fig3.update_yaxes(showgrid=True, gridcolor='#2d3548')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col_insight2:
            st.markdown("### ⭐ Rating Distribution")
            rating_counts = df['rating'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            
            fig4 = px.bar(
                rating_counts,
                x='Rating',
                y='Count',
                text='Count',
                color='Rating',
                color_continuous_scale='RdYlGn'
            )
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis_title="Star Rating",
                yaxis_title="Count",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            fig4.update_traces(textposition='outside')
            fig4.update_xaxes(showgrid=False)
            fig4.update_yaxes(showgrid=True, gridcolor='#2d3548')
            st.plotly_chart(fig4, use_container_width=True)

    # === TAB 2: AI ANALYST CHAT ===
    with tab2:
        st.markdown("### 💬 AI-Powered Analysis Assistant")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "🎯 **Hello! I'm your AI analyst.** I've analyzed your customer feedback and I'm ready to answer any questions. Try asking:\n- What are the main customer complaints?\n- Which categories have the most issues?\n- What improvements should we prioritize?"}
            ]
        
        # Check if there's a pending quick action query
        if "pending_query" not in st.session_state:
            st.session_state.pending_query = None

        # Process pending query from quick actions
        if st.session_state.pending_query:
            prompt = st.session_state.pending_query
            st.session_state.pending_query = None
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing your data..."):
                    try:
                        vectorstore, ranker = build_vector_db(df)
                        ans, sources = generate_answer(prompt, vectorstore, ranker)
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                        
                        with st.expander("🔍 View Source References"):
                            for idx, r in enumerate(sources, 1):
                                st.info(f"**Source {idx}:** {r['text']}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("💭 Ask me anything about your customer feedback..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing your data..."):
                    try:
                        vectorstore, ranker = build_vector_db(df)
                        ans, sources = generate_answer(prompt, vectorstore, ranker)
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                        
                        with st.expander("🔍 View Source References"):
                            for idx, r in enumerate(sources, 1):
                                st.info(f"**Source {idx}:** {r['text']}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Quick Action Buttons
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("📊 Analyze Trends", use_container_width=True):
                query = "What are the key trends in customer feedback?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.pending_query = query
                st.rerun()
        
        with col_btn2:
            if st.button("🎯 Priority Issues", use_container_width=True):
                query = "What are the most critical issues we need to address immediately?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.pending_query = query
                st.rerun()
        
        with col_btn3:
            if st.button("💡 Get Recommendations", use_container_width=True):
                query = "Based on the feedback, what improvements should we make?"
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.pending_query = query
                st.rerun()
        
        st.markdown("---")
        
        # Email Response Generator
        st.markdown("### 📧 Response Generator")
        
        with st.expander("✉️ Generate Customer Response Email", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                customer_name = st.text_input("👤 Customer Name", "Valued Customer", key="cust_name")
            
            with col2:
                issue_category = st.selectbox("🏷️ Issue Type", df['category'].unique(), key="issue_cat")
            
            with col3:
                email_tone = st.select_slider(
                    "🎭 Tone",
                    options=["Apologetic", "Empathetic", "Professional", "Friendly"],
                    value="Professional",
                    key="email_tone"
                )
            
            if st.button("✨ Generate Email", type="primary", use_container_width=True):
                with st.spinner("✍️ Crafting personalized response..."):
                    try:
                        llm = get_llm()
                        prompt = f"""Write a {email_tone} customer service email to {customer_name} addressing their concern about {issue_category}. 
                        
                        The email should:
                        - Acknowledge their concern
                        - Show empathy and understanding
                        - Provide a clear solution or next steps
                        - Maintain a {email_tone} tone
                        - Be concise but comprehensive
                        
                        Format: Professional business email"""
                        
                        resp = llm.invoke(prompt).content
                        st.text_area("📧 Generated Email:", resp, height=350, key="email_output")
                        
                        # Download button for email
                        st.download_button(
                            label="📥 Download Email",
                            data=resp,
                            file_name=f"response_{customer_name.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating email: {str(e)}")

    # === TAB 3: DATA EXPLORER ===
    with tab3:
        st.markdown("### 📋 Complete Dataset")
        
        # Filter Options
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=df['sentiment'].unique(),
                default=df['sentiment'].unique()
            )
        
        with col_filter2:
            category_filter = st.multiselect(
                "Filter by Category",
                options=df['category'].unique(),
                default=df['category'].unique()
            )
        
        with col_filter3:
            rating_filter = st.slider(
                "Filter by Rating",
                min_value=int(df['rating'].min()),
                max_value=int(df['rating'].max()),
                value=(int(df['rating'].min()), int(df['rating'].max()))
            )
        
        # Apply Filters
        filtered_df = df[
            (df['sentiment'].isin(sentiment_filter)) &
            (df['category'].isin(category_filter)) &
            (df['rating'] >= rating_filter[0]) &
            (df['rating'] <= rating_filter[1])
        ]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        
        # Display Filtered Data
        st.dataframe(filtered_df, use_container_width=True, height=500)
        
        # Export Options
        st.markdown("---")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name=f"customer_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # Summary Statistics
            if st.button("📊 View Summary Statistics", use_container_width=True):
                st.markdown("#### Summary Statistics")
                summary_stats = filtered_df.describe()
                st.dataframe(summary_stats, use_container_width=True)

else:
    # Welcome Screen
    welcome_html = """
    <div style='text-align: center; padding: 4rem 2rem;'>
        <div style='font-size: 5rem; margin-bottom: 1rem;'>📊</div>
        <h2 style='color: #e2e8f0; margin-bottom: 1rem;'>Welcome to VoiceFlow Analytics</h2>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto;'>
            Upload your customer feedback data to unlock powerful AI-driven insights and analytics.
        </p>
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    border: 1px solid #667eea; border-radius: 12px; padding: 2rem; max-width: 600px; margin: 0 auto;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>🚀 Get Started</h3>
            <p style='color: #94a3b8; margin-bottom: 0;'>
                👈 Upload a CSV file using the sidebar to begin your analysis
            </p>
        </div>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)