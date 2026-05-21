import os
import streamlit as st
import requests
from dotenv import load_dotenv

# Load env variables
load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DOCUMENTS_DIR = os.path.join(_PROJECT_ROOT, "documents")
CHROMA_DB_PATH = os.path.join(_PROJECT_ROOT, "app", "database", "chroma_db")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api")

# Try importing backend directly for standalone runtime support
try:
    from app.rag.qa_chain import query_rag
    from app.rag.contradiction import analyze_contradiction
    from app.rag.ingest import run_ingestion
    LOCAL_IMPORTS_AVAILABLE = True
except Exception as e:
    LOCAL_IMPORTS_AVAILABLE = False
    LOCAL_IMPORT_ERROR = str(e)

# Streamlit App Configurations
st.set_page_config(
    page_title="RefineRAG - Citation Document Q&A & Audit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4B8BFF, #FF7676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.8rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    
    .citation-card {
        background: rgba(75, 139, 255, 0.04);
        border-left: 4px solid #4B8BFF;
        border-radius: 4px 12px 12px 4px;
        padding: 15px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .citation-meta {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4B8BFF;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
    }
    
    .citation-snippet {
        font-style: italic;
        color: #E2E8F0;
        line-height: 1.4;
    }
    
    .contradiction-header {
        background: linear-gradient(90deg, #E53E3E, #FC8181);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
        box-shadow: 0 4px 14px 0 rgba(229, 62, 62, 0.2);
    }
    
    .no-contradiction-header {
        background: linear-gradient(90deg, #38A169, #68D391);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
        box-shadow: 0 4px 14px 0 rgba(56, 161, 105, 0.2);
    }
    
    .evidence-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        height: 100%;
    }
    
    .evidence-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #CBD5E0;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .evidence-text {
        font-size: 0.9rem;
        color: #E2E8F0;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.markdown("<div class='main-title'>RefineRAG</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-grade Document Q&A with Strict Page-Level Citations & Contradiction Auditing</div>", unsafe_allow_html=True)

# Check folders
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Helper to list current files (excluding readme)
def get_uploaded_files():
    files = []
    for f in os.listdir(DOCUMENTS_DIR):
        if os.path.isfile(os.path.join(DOCUMENTS_DIR, f)) and f.lower() != "readme.md":
            files.append(f)
    return files

# Sidebar: Document Management
with st.sidebar:
    st.header("📚 Document Ingestion")
    
    # Upload new file
    uploaded_file = st.file_uploader("Upload PDF, TXT or MD", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        st.rerun()

    # List files
    current_files = get_uploaded_files()
    if current_files:
        st.write("---")
        st.write("**Ingested Corpus Files:**")
        for idx, file in enumerate(current_files):
            col1, col2 = st.columns([0.85, 0.15])
            col1.text(f"📄 {file}")
            if col2.button("🗑️", key=f"del_{idx}"):
                os.remove(os.path.join(DOCUMENTS_DIR, file))
                st.success(f"Removed {file}")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

    # Re-indexing Button
    st.write("---")
    st.write("**Database Controls:**")
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Processing documents, generating embeddings and rebuilding index..."):
            if LOCAL_IMPORTS_AVAILABLE:
                count, msg = run_ingestion(DOCUMENTS_DIR, CHROMA_DB_PATH)
                if count > 0:
                    st.success(msg)
                else:
                    st.warning(msg)
            else:
                st.error("Ingestion module is unavailable locally.")

# Tab setup
tab_qa, tab_contradict = st.tabs(["💬 Document Q&A Engine", "⚖️ Document Contradiction Auditor"])

# Tab 1: Q&A Engine
with tab_qa:
    st.write("Ask questions based on your ingested documents. The AI is restricted to the provided context and will provide citations.")
    
    col_input, col_lang = st.columns([0.75, 0.25])
    
    with col_input:
        query = st.text_input("Enter your question:", placeholder="e.g. What is the stipend amount in v1?")
        
    with col_lang:
        lang_options = ["Original (Auto)", "English", "Spanish", "French", "German", "Hindi", "Gujarati", "Chinese"]
        selected_lang = st.selectbox("Response Language:", options=lang_options)

    if st.button("🚀 Answer Question", type="primary"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing document database..."):
                target_lang_param = None if selected_lang == "Original (Auto)" else selected_lang
                
                # Setup execution: Try calling FastAPI API server, fallback to Direct python imports
                result = None
                using_api = False
                
                try:
                    payload = {"question": query}
                    if target_lang_param:
                        payload["language"] = target_lang_param
                        
                    response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=15)
                    if response.status_code == 200:
                        result = response.json()
                        using_api = True
                except Exception as api_err:
                    # Fallback to local import execution
                    pass

                if not result and LOCAL_IMPORTS_AVAILABLE:
                    try:
                        local_res = query_rag(query, target_lang=target_lang_param)
                        result = {
                            "answer": local_res["answer"],
                            "citations": local_res["citations"],
                            "confidence": local_res["confidence"]
                        }
                    except Exception as local_err:
                        st.error(f"Local query execution failed: {str(local_err)}")
                elif not result:
                    st.error("Could not reach API server and local modules are missing.")

                if result:
                    # Print Answer Card
                    st.markdown("### Answer")
                    
                    # Style confidence pill
                    conf = result["confidence"]
                    conf_class = "confidence-high" if conf >= 0.7 else "confidence-low"
                    conf_label = f"Confidence: {int(conf * 100)}%" if result["answer"] != "Not found in the provided documents." else "Not Found"
                    
                    st.markdown(f"""
                    <div class="card">
                        <div style="font-size: 1.15rem; line-height: 1.6; margin-bottom: 12px;">{result["answer"]}</div>
                        <div><span class="confidence-badge {conf_class}">{conf_label}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Print Citations
                    if result["citations"]:
                        st.markdown("### Source Citations")
                        for cit in result["citations"]:
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-meta">
                                    <span>📄 {cit['file']} (Page {cit['page']})</span>
                                    <span>Chunk ID: {cit['chunk_id']}</span>
                                </div>
                                <div class="citation-snippet">"{cit['snippet']}"</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        if result["answer"] != "Not found in the provided documents.":
                            st.info("No precise source citation could be mapped to the response.")

# Tab 2: Contradiction Auditor
with tab_contradict:
    st.write("Compare any two documents on a specific topic to locate factual discrepancies, outdated policies, or conflicting details.")
    
    files = get_uploaded_files()
    if len(files) < 2:
        st.warning("Please upload at least 2 documents to compare versions and check contradictions.")
    else:
        col_doc1, col_doc2, col_topic = st.columns([0.3, 0.3, 0.4])
        
        with col_doc1:
            doc1 = st.selectbox("Document 1:", options=files, index=0)
            
        with col_doc2:
            # Try to pick second file as default if possible
            default_idx = 1 if len(files) > 1 else 0
            doc2 = st.selectbox("Document 2:", options=files, index=default_idx)
            
        with col_topic:
            topic = st.text_input("Audited Topic (e.g. stipend, duration):", placeholder="e.g. stipend")

        if st.button("⚖️ Run Version Audit", type="secondary"):
            if doc1 == doc2:
                st.warning("Please select two different documents to perform comparison.")
            elif not topic.strip():
                st.warning("Please specify a topic to inspect.")
            else:
                with st.spinner(f"Auditing '{topic}' between {doc1} and {doc2}..."):
                    result = None
                    using_api = False
                    
                    # Try calling FastAPI `/contradict`
                    try:
                        payload = {"doc1_id": doc1, "doc2_id": doc2, "topic": topic}
                        response = requests.post(f"{API_BASE_URL}/contradict", json=payload, timeout=20)
                        if response.status_code == 200:
                            result = response.json()
                            using_api = True
                    except Exception:
                        pass
                        
                    # Fallback to local execution
                    if not result and LOCAL_IMPORTS_AVAILABLE:
                        try:
                            local_res = analyze_contradiction(doc1, doc2, topic)
                            result = local_res
                        except Exception as e:
                            st.error(f"Local audit failed: {str(e)}")
                    elif not result:
                        st.error("Could not reach API server and local modules are missing.")

                    if result:
                        is_conflict = result["contradiction_found"]
                        
                        # Warning/Success Banner
                        if is_conflict:
                            st.markdown(f"""
                            <div class="contradiction-header">
                                🚨 CONTRADICTION DETECTED on topic "{topic}"
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="no-contradiction-header">
                                ✅ NO CONTRADICTIONS DETECTED on topic "{topic}"
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Reasoning Card
                        st.markdown("#### AI Auditor Reasoning")
                        st.markdown(f"""
                        <div class="card" style="font-size: 1.05rem; line-height: 1.5;">
                            {result["reasoning"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Evidence side-by-side Columns
                        col_ev1, col_ev2 = st.columns(2)
                        
                        with col_ev1:
                            st.markdown(f"""
                            <div class="evidence-box">
                                <div class="evidence-title">Snippet from {doc1}</div>
                                <div class="evidence-text">"{result['evidence_doc1'] if result['evidence_doc1'] else 'No specific evidence located.'}"</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col_ev2:
                            st.markdown(f"""
                            <div class="evidence-box">
                                <div class="evidence-title">Snippet from {doc2}</div>
                                <div class="evidence-text">"{result['evidence_doc2'] if result['evidence_doc2'] else 'No specific evidence located.'}"</div>
                            </div>
                            """, unsafe_allow_html=True)
