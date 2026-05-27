import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background:
    radial-gradient(circle at top left, #1E3A8A, transparent 30%),
    radial-gradient(circle at bottom right, #7C3AED, transparent 30%),
    linear-gradient(135deg, #050816, #0B1026, #111936);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

section[data-testid="stSidebar"] {
    background: rgba(10,15,35,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.hero {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 30px;
    padding: 3rem;
    backdrop-filter: blur(16px);
    margin-bottom: 2rem;
    box-shadow: 0 0 40px rgba(124,58,237,0.25);
}

.title {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #60A5FA,
        #A855F7,
        #EC4899
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #CBD5E1;
    font-size: 1.1rem;
    margin-top: 1rem;
}

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 30px rgba(168,85,247,0.25);
}

.upload-box {
    background: rgba(255,255,255,0.03);
    border: 2px dashed #8B5CF6;
    border-radius: 30px;
    padding: 3rem;
    text-align: center;
    margin-top: 2rem;
}

.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 12px !important;
}

.stButton>button {
    background: linear-gradient(
        90deg,
        #7C3AED,
        #9333EA
    );
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 2rem;
    font-size: 1rem;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("""
    <h1 style='color:white;'>🧠 AI Research</h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📄 Upload Papers",
            "🔍 Semantic Search",
            "🤖 AI Assistant",
            "📚 Vector Database",
            "⚙️ Settings"
        ]
    )

    st.markdown("---")

    st.markdown("""
    ### 🚀 System Status

    ✅ Embeddings Ready  
    ✅ FAISS Active  
    ✅ RAG Pipeline Online  
    ✅ Groq Connected
    """)

    st.markdown("---")

    st.markdown("""
    ### 🛠 Tech Stack

    - LangChain
    - FAISS
    - HuggingFace
    - Groq LLM
    - Streamlit
    """)

st.markdown("""
<div class="hero">

<div class="title">
🧠 AI Research Assistant
</div>

<div class="subtitle">
Next-generation AI platform for understanding
research papers using Retrieval-Augmented Generation,
semantic vector search, transformer embeddings,
and large language models.
</div>

<br>

<div style="
display:flex;
gap:10px;
flex-wrap:wrap;
">

<div style="
background:rgba(255,255,255,0.08);
padding:10px 18px;
border-radius:999px;
">
⚡ RAG Pipeline
</div>

<div style="
background:rgba(255,255,255,0.08);
padding:10px 18px;
border-radius:999px;
">
🔍 Semantic Search
</div>

<div style="
background:rgba(255,255,255,0.08);
padding:10px 18px;
border-radius:999px;
">
📚 FAISS Vector DB
</div>

<div style="
background:rgba(255,255,255,0.08);
padding:10px 18px;
border-radius:999px;
">
🤖 LLM Powered
</div>

</div>

</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

chunks = []

if uploaded_files:

    all_text = ""

    for file in uploaded_files:

        pdf_reader = PdfReader(file)

        text = ""

        for page in pdf_reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        all_text += text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(all_text)

doc_count = len(uploaded_files) if uploaded_files else 0
chunk_count = len(chunks)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="card">
    <h3>📄 Documents</h3>
    <h1>{doc_count}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
    <h3>🧠 Embeddings</h3>
    <h1>{chunk_count}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>⚡ Queries</h3>
    <h1>Live</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
    <h3>🤖 Status</h3>
    <h1>Ready</h1>
    </div>
    """, unsafe_allow_html=True)

if selected == "🏠 Dashboard":

    st.subheader("🏠 Dashboard")

    st.markdown("""
    <div class="card">
    Welcome to your AI-powered research workspace.
    Upload papers, perform semantic search,
    and chat with your documents using RAG.
    </div>
    """, unsafe_allow_html=True)

elif selected == "📄 Upload Papers":

    st.subheader("📄 Upload Research Papers")

    st.markdown("""
    <div class="upload-box">
    <h1>📤 Upload Research Papers</h1>

    <p style='font-size:18px;color:#CBD5E1;'>
    Upload PDFs and build your own intelligent AI knowledge base
    </p>

    </div>
    """, unsafe_allow_html=True)

    if uploaded_files:

        st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded")

        for file in uploaded_files:

            st.markdown(f"""
            <div class="card">
            📄 {file.name}
            </div>
            """, unsafe_allow_html=True)

elif selected == "🔍 Semantic Search":

    st.subheader("🔍 Semantic Search")

    if uploaded_files:

        st.write(f"Total Chunks Created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:5]):

            st.markdown(f"""
            <div class="card">
            <h3>Chunk {i+1}</h3>
            <p>{chunk[:500]}...</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload PDFs first.")

elif selected == "🤖 AI Assistant":

    st.subheader("🤖 AI Assistant")

    if uploaded_files:

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_texts(
            chunks,
            embedding=embedding_model
        )

        query = st.text_input(
            "Ask anything about the research paper"
        )

        if query:

            docs = vector_store.similarity_search(
                query,
                k=3
            )

            llm = ChatGroq(
                temperature=0,
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are an AI Research Assistant.

Answer the question ONLY using the provided context.

Research Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt).content

            st.markdown(f"""
            <div class="card">
            <h2>🤖 AI Answer</h2>
            <p>{response}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload PDFs first.")

elif selected == "📚 Vector Database":

    st.subheader("📚 Vector Database")

    st.markdown(f"""
    <div class="card">
    <h2>FAISS Vector Store</h2>

    <p>Total Document Chunks: {chunk_count}</p>

    <p>Embedding Model:</p>

    <code>sentence-transformers/all-MiniLM-L6-v2</code>
    </div>
    """, unsafe_allow_html=True)

elif selected == "⚙️ Settings":

    st.subheader("⚙️ Settings")

    st.markdown("""
    <div class="card">
    <h3>Application Settings</h3>

    <p>Theme: Dark Neon</p>

    <p>LLM Provider: Groq</p>

    <p>Vector Database: FAISS</p>

    <p>Status: Operational</p>

    </div>
    """, unsafe_allow_html=True)