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

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.markdown("""
<style>

.stApp {
background:
radial-gradient(circle at top left,#1E3A8A,transparent 30%),
radial-gradient(circle at bottom right,#7C3AED,transparent 30%),
linear-gradient(135deg,#050816,#0B1026,#111936);
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

section[data-testid="stSidebar"]{
background:rgba(10,15,35,0.95);
}

.card{
background:rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.08);
border-radius:20px;
padding:20px;
margin-bottom:15px;
}

.hero{
background:rgba(255,255,255,0.05);
padding:35px;
border-radius:25px;
margin-bottom:20px;
}

.stButton button{
background:#7C3AED;
color:white;
border:none;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.title("🧠 AI Research")

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
    st.markdown("""
<div class="hero">
<h1>🧠 AI Research Assistant</h1>
<p>
Upload research papers, create embeddings,
perform semantic search, and chat with your documents.
</p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload PDF Research Papers",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.session_state.uploaded_files = uploaded_files

    all_text = ""

    for file in uploaded_files:

        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                all_text += text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(all_text)

    st.session_state.chunks = chunks

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding_model
    )

    st.session_state.vector_store = vector_store

uploaded_files = st.session_state.uploaded_files
chunks = st.session_state.chunks

doc_count = len(uploaded_files) if uploaded_files else 0
chunk_count = len(chunks)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 Documents", doc_count)

with col2:
    st.metric("🧠 Chunks", chunk_count)

with col3:
    st.metric("⚡ Status", "Ready")

with col4:
    st.metric("🤖 AI", "Online")
    if selected == "🏠 Dashboard":

    st.subheader("🏠 Dashboard")

    st.markdown("""
    <div class="card">
    <h3>Welcome</h3>
    <p>
    This AI Research Assistant helps you upload papers,
    perform semantic search, generate embeddings,
    and chat with research documents.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_files:

        st.success(f"{len(uploaded_files)} document(s) loaded successfully.")

        for file in uploaded_files:

            st.write("📄", file.name)

    else:

        st.info("Upload research papers to begin.")

elif selected == "📄 Upload Papers":

    st.subheader("📄 Upload Papers")

    if uploaded_files:

        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

        for file in uploaded_files:

            st.markdown(f"""
            <div class="card">
            <h4>📄 {file.name}</h4>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.warning("No PDFs uploaded yet.")

elif selected == "🔍 Semantic Search":

    st.subheader("🔍 Semantic Search")

    if chunks:

        st.write(f"Total Chunks Created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:5]):

            st.markdown(f"""
            <div class="card">
            <h4>Chunk {i+1}</h4>
            <p>{chunk[:400]}...</p>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.info("Upload PDFs first.")

elif selected == "🤖 AI Assistant":

    st.subheader("🤖 AI Assistant")

    if st.session_state.vector_store:

        question = st.chat_input(
            "Ask anything about your uploaded papers..."
        )

        if question:

            with st.chat_message("user"):
                st.write(question)

            docs = st.session_state.vector_store.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-8b-8192",
                temperature=0
            )

            prompt = f"""
You are an AI Research Assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = llm.invoke(prompt).content

            with st.chat_message("assistant"):
                st.write(answer)

    else:

        st.info("Upload PDFs first.")

elif selected == "📚 Vector Database":

    st.subheader("📚 Vector Database")

    st.markdown(f"""
    <div class="card">
    <h3>FAISS Vector Store</h3>
    <p>Total Chunks: {chunk_count}</p>
    <p>Embedding Model:</p>
    <code>sentence-transformers/all-MiniLM-L6-v2</code>
    </div>
    """, unsafe_allow_html=True)
    if selected == "🏠 Dashboard":

    st.subheader("🏠 Dashboard")

    st.markdown("""
    <div class="card">
    <h3>Welcome</h3>
    <p>
    This AI Research Assistant helps you upload papers,
    perform semantic search, generate embeddings,
    and chat with research documents.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_files:

        st.success(f"{len(uploaded_files)} document(s) loaded successfully.")

        for file in uploaded_files:

            st.write("📄", file.name)

    else:

        st.info("Upload research papers to begin.")

elif selected == "📄 Upload Papers":

    st.subheader("📄 Upload Papers")

    if uploaded_files:

        st.success(f"{len(uploaded_files)} PDF(s) uploaded")

        for file in uploaded_files:

            st.markdown(f"""
            <div class="card">
            <h4>📄 {file.name}</h4>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.warning("No PDFs uploaded yet.")

elif selected == "🔍 Semantic Search":

    st.subheader("🔍 Semantic Search")

    if chunks:

        st.write(f"Total Chunks Created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:5]):

            st.markdown(f"""
            <div class="card">
            <h4>Chunk {i+1}</h4>
            <p>{chunk[:400]}...</p>
            </div>
            """, unsafe_allow_html=True)

    else:

        st.info("Upload PDFs first.")

elif selected == "🤖 AI Assistant":

    st.subheader("🤖 AI Assistant")

    if st.session_state.vector_store:

        question = st.chat_input(
            "Ask anything about your uploaded papers..."
        )

        if question:

            with st.chat_message("user"):
                st.write(question)

            docs = st.session_state.vector_store.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-8b-8192",
                temperature=0
            )

            prompt = f"""
You are an AI Research Assistant.

Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = llm.invoke(prompt).content

            with st.chat_message("assistant"):
                st.write(answer)

    else:

        st.info("Upload PDFs first.")

elif selected == "📚 Vector Database":

    st.subheader("📚 Vector Database")

    st.markdown(f"""
    <div class="card">
    <h3>FAISS Vector Store</h3>
    <p>Total Chunks: {chunk_count}</p>
    <p>Embedding Model:</p>
    <code>sentence-transformers/all-MiniLM-L6-v2</code>
    </div>
    """, unsafe_allow_html=True)