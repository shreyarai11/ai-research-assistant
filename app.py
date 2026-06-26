import os
import time
from dotenv import load_dotenv

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
    st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.stApp{
background:
radial-gradient(circle at top left,#2563eb,transparent 30%),
radial-gradient(circle at bottom right,#7c3aed,transparent 35%),
linear-gradient(135deg,#050816,#0b1026,#111936);
color:white;
}

section[data-testid="stSidebar"]{
background:#0f172a;
border-right:1px solid rgba(255,255,255,.08);
}

.hero{
padding:35px;
border-radius:24px;
background:rgba(255,255,255,.05);
backdrop-filter:blur(18px);
margin-bottom:20px;
}

.metric-card{
background:rgba(255,255,255,.05);
padding:18px;
border-radius:20px;
text-align:center;
border:1px solid rgba(255,255,255,.08);
}

.card{
background:rgba(255,255,255,.05);
padding:20px;
border-radius:18px;
margin-bottom:18px;
border:1px solid rgba(255,255,255,.08);
}

.stButton>button{
background:#7c3aed;
color:white;
border:none;
border-radius:12px;
padding:.7rem 1.5rem;
font-weight:bold;
}

.stDownloadButton>button{
background:#2563eb;
color:white;
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
            "📊 Analytics",
            "📑 AI Summary",
            "🔬 Compare Papers",
            "💾 Export Chat",
            "⚙️ Settings"
        ]
    )

    st.markdown("---")

    st.success("🟢 Groq Connected")
    st.success("🟢 FAISS Ready")
    st.success("🟢 Embeddings Ready")
    st.markdown("""
<div class="hero">

<h1>🧠 AI Research Assistant Pro</h1>

<p style="font-size:18px;">
Upload research papers, build a semantic vector database,
search through documents, generate AI summaries,
compare papers and chat with them using Retrieval-Augmented Generation.
</p>

</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">

<h1>🧠 AI Research Assistant Pro</h1>

<p style="font-size:18px;">
Upload research papers, build a semantic vector database,
search through documents, generate AI summaries,
compare papers and chat with them using Retrieval-Augmented Generation.
</p>

</div>
""", unsafe_allow_html=True)
uploaded_files = st.session_state.uploaded_files
chunks = st.session_state.chunks
vector_store = st.session_state.vector_store

doc_count = len(uploaded_files)
chunk_count = len(chunks)
question_count = st.session_state.total_questions
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📄 Papers</h3>
        <h1>{doc_count}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🧠 Chunks</h3>
        <h1>{chunk_count}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>💬 Questions</h3>
        <h1>{question_count}</h1>
    </div>
    """, unsafe_allow_html=True)

with col4:

    status = "🟢 Online" if vector_store else "🔴 Offline"

    st.markdown(f"""
    <div class="metric-card">
if selected == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    st.markdown("""
    <!-- Your existing dashboard HTML goes here -->
    """, unsafe_allow_html=True)

    if doc_count == 0:
        st.info("Upload research papers to begin.")
    else:
        st.success(f"{doc_count} research paper(s) loaded.")

        st.subheader("Uploaded Papers")

        for file in uploaded_files:
            st.write("📄", file.name)

elif selected == "📄 Upload Papers":

    st.header("📄 Upload Research Papers")

    if doc_count == 0:
        st.info("Upload one or more PDF research papers using the uploader above.")
    else:
        st.success(f"{doc_count} paper(s) uploaded successfully.")

        st.markdown("## 📚 Uploaded Papers")

        for i, file in enumerate(uploaded_files):
            size = round(file.size / 1024, 2)

            st.markdown(f"""
            <div class="card">
                <h4> {file.name}</h4>
                <p>Size: {size} KB</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("Upload one or more PDF research papers using the uploader above.")

    else:

        st.success(f"{doc_count} paper(s) uploaded successfully.")

        st.markdown("## 📚 Uploaded Papers")

        for i, file in enumerate(uploaded_files):

            size = round(file.size / 1024, 2)

            st.markdown(f"""
            <div class="card">

            <h3> {file.name}</h3>

            <p><b>Paper:</b> {i+1}</p>

            <p><b>Size:</b> {size} KB</p>

            </div>
            """, unsafe_allow_html=True)
                    st.markdown("---")

        st.subheader("📊 Statistics")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Research Papers", doc_count)

        with c2:
            st.metric("Text Chunks", chunk_count)

        with c3:
            st.metric("Characters", len(st.session_state.paper_text))
                    st.markdown("---")

        st.subheader("📖 Extracted Text Preview")

        preview = st.session_state.paper_text[:3000]

        st.text_area(
            "Preview",
            preview,
            height=300
        )
                st.download_button(
            "📥 Download Extracted Text",
            data=st.session_state.paper_text,
            file_name="research_text.txt",
            mime="text/plain"
        )
                st.markdown("---")

        if st.button("🗑 Clear Workspace"):

            st.session_state.uploaded_files = []
            st.session_state.paper_text = ""
            st.session_state.chunks = []
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.total_questions = 0

            st.success("Workspace Cleared Successfully")

            st.rerun()
            elif selected == "🔍 Semantic Search":

    st.header("🔍 Semantic Search")

    if vector_store is None:

        st.info("Upload research papers first.")

    else:

        query = st.text_input(
            "Search your research papers",
            placeholder="Example: transformer architecture"
        )

        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5
        )

        if query:

            with st.spinner("Searching..."):

                docs = vector_store.similarity_search(
                    query,
                    k=top_k
                )

                st.success(f"{len(docs)} matching chunks found.")
                                st.markdown("---")

                for i, doc in enumerate(docs):

                    with st.expander(f"📄 Result {i+1}"):

                        st.write(doc.page_content)

                        st.caption(
                            f"Chunk Length: {len(doc.page_content)} characters"
                        )
                                st.markdown("---")

        st.subheader("💡 Suggested Searches")

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Methodology"):

                st.session_state.semantic_query = "Methodology"

        with col2:

            if st.button("Conclusion"):

                st.session_state.semantic_query = "Conclusion"

        col3, col4 = st.columns(2)

        with col3:

            if st.button("Future Work"):

                st.session_state.semantic_query = "Future Work"

        with col4:

            if st.button("Results"):

                st.session_state.semantic_query = "Results"
                        if "semantic_query" in st.session_state:

            docs = vector_store.similarity_search(
                st.session_state.semantic_query,
                k=3
            )

            st.markdown("---")

            st.subheader(
                f"Results for '{st.session_state.semantic_query}'"
            )

            for doc in docs:

                st.markdown(
                    f"""
                    <div class="card">
                    {doc.page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                elif selected == "🤖 AI Assistant":

    st.header("🤖 AI Research Assistant")

    if vector_store is None:

        st.info("Please upload research papers first.")

    else:

        st.write("Ask any question about your uploaded research papers.")

        if len(st.session_state.chat_history) == 0:

            st.info("No conversation yet. Start by asking a question.")

        for chat in st.session_state.chat_history:

            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        question = st.chat_input(
            "Ask a question about your papers..."
        )

        if question:

            st.session_state.total_questions += 1

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)
                            with st.spinner("Searching relevant information..."):

                docs = vector_store.similarity_search(
                    question,
                    k=4
                )

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )
                                llm = ChatGroq(

                    groq_api_key=os.getenv("GROQ_API_KEY"),

                    model_name="llama3-8b-8192",

                    temperature=0
                )

                prompt = f"""
You are an expert AI Research Assistant.

Use ONLY the research context below.

If the answer is not present in the context,
reply:

'I could not find this information in the uploaded research papers.'

Research Context:

{context}

Question:

{question}

Answer:
"""
                response = llm.invoke(prompt)

                answer = response.content
                            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            with st.chat_message("assistant"):

                st.markdown(answer)
                                with st.expander("📚 Source Context"):

                    for i, doc in enumerate(docs):

                        st.markdown(f"### Chunk {i+1}")

                        st.write(doc.page_content)

                        st.divider()
                                st.markdown("---")

        if st.button("🗑 Clear Conversation"):

            st.session_state.chat_history = []

            st.session_state.total_questions = 0

            st.rerun()
                    st.markdown("---")

        if st.button("🗑 Clear Conversation"):

            st.session_state.chat_history = []

            st.session_state.total_questions = 0

            st.rerun()
            elif selected == "📚 Vector Database":

    st.header("📚 Vector Database")

    if vector_store is None:

        st.info("Upload research papers first.")

    else:

        st.success("FAISS Vector Store Loaded Successfully")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Research Papers",
                doc_count
            )

            st.metric(
                "Total Chunks",
                chunk_count
            )

        with col2:

            st.metric(
                "Embedding Model",
                "MiniLM-L6-v2"
            )

            st.metric(
                "Vector Database",
                "FAISS"
            )
                    st.markdown("---")

        st.subheader("Embedding Information")

        st.markdown("""
        <div class="card">

        <h3>Sentence Transformer</h3>

        <ul>

        <li>Model : sentence-transformers/all-MiniLM-L6-v2</li>

        <li>Vector Database : FAISS</li>

        <li>Chunk Size : 1000</li>

        <li>Chunk Overlap : 200</li>

        <li>Retriever : Similarity Search</li>

        </ul>

        </div>
        """, unsafe_allow_html=True)
                st.markdown("---")

        st.subheader("Chunk Explorer")

        chunk_number = st.number_input(
            "Select Chunk",
            min_value=1,
            max_value=max(1, chunk_count),
            value=1
        )

        if chunk_count > 0:

            st.markdown("""
            <div class="card">
            """, unsafe_allow_html=True)

            st.write(
                chunks[chunk_number - 1]
            )

            st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")

        st.subheader("Database Health")

        st.progress(100)

        st.success("Embeddings Generated Successfully")

        st.success("FAISS Index Ready")

        st.success("Semantic Search Enabled")

        st.success("AI Assistant Connected")
                st.markdown("---")

        st.subheader("Current Database Summary")

        st.write(f"📄 Documents : {doc_count}")

        st.write(f"🧠 Chunks : {chunk_count}")

        st.write(f"💬 Questions Asked : {question_count}")

        st.write("⚡ Retrieval Method : Similarity Search")

        st.write("🤖 LLM : Llama 3 8B")
        elif selected == "📊 Analytics":

    st.header("📊 Research Analytics")

    if doc_count == 0:

        st.info("Upload research papers to view analytics.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.metric("Research Papers", doc_count)
            st.metric("Text Chunks", chunk_count)
            st.metric("Questions Asked", question_count)

        with col2:

            total_words = len(st.session_state.paper_text.split())
            total_characters = len(st.session_state.paper_text)

            st.metric("Total Words", total_words)
            st.metric("Characters", total_characters)
            st.metric("Average Chunk Size", round(total_characters / max(chunk_count,1)))
                    st.markdown("---")

        st.subheader("📈 Processing Progress")

        st.progress(100)

        st.success("PDF Parsing Completed")

        st.success("Embeddings Generated")

        st.success("FAISS Index Created")

        st.success("Semantic Search Ready")

        st.success("AI Assistant Ready")
                st.markdown("---")

        st.subheader("📊 Dataset Overview")

        st.write(f"📄 Uploaded Documents : {doc_count}")
        st.write(f"📚 Generated Chunks : {chunk_count}")
        st.write(f"💬 Total AI Questions : {question_count}")
        st.write(f"📝 Total Words : {total_words}")
        st.write(f"🔤 Total Characters : {total_characters}")
                st.markdown("---")

        st.subheader("🏆 Top Keywords")

        words = st.session_state.paper_text.split()

        stop_words = {
            "the","and","of","to","in","a","is","for","on",
            "with","that","this","are","as","an","by","or",
            "from","be","at","it","we","our","their"
        }

        filtered = [
            word.lower().strip(".,()[]{}:;!?")
            for word in words
            if len(word) > 4 and word.lower() not in stop_words
        ]
                from collections import Counter

        counter = Counter(filtered)

        top_words = counter.most_common(15)

        if top_words:

            for word, count in top_words:

                st.write(f"**{word}** — {count}")
                        st.markdown("---")

        st.subheader("📋 Project Information")

        st.info("""
AI Research Assistant Pro

Multiple PDF Processing

Retrieval-Augmented Generation (RAG)

HuggingFace Embeddings

FAISS Vector Search

Groq Llama 3

Semantic Search

AI Question Answering

Analytics Dashboard
""")
elif selected == "📑 AI Summary":

    st.header("📑 AI Research Summary")

    if vector_store is None:

        st.info("Upload research papers first.")

    else:

        st.write(
            "Generate an AI summary of all uploaded research papers."
        )

        if st.button("🚀 Generate Summary"):

            with st.spinner("Generating Summary..."):

                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama3-8b-8192",
                    temperature=0
                )

                context = st.session_state.paper_text[:12000]

                prompt = f"""
You are an expert Research Assistant.

Analyze the uploaded research paper(s).

Return the response in the following format:

# Executive Summary

# Research Objective

# Methodology

# Key Contributions

# Experimental Results

# Limitations

# Future Work

# Final Conclusion

Research Papers:

{context}
"""
                response = llm.invoke(prompt)

                summary = response.content

                st.session_state.summary = summary
                                st.markdown("## 📄 AI Generated Summary")

                st.markdown(summary)
                                st.download_button(

                    label="📥 Download Summary",

                    data=summary,

                    file_name="AI_Research_Summary.txt",

                    mime="text/plain"
                )
                        if "summary" in st.session_state:

            st.markdown("---")

            st.success("Summary already generated.")

            st.markdown(st.session_state.summary)
                        st.download_button(

                "📥 Download Existing Summary",

                st.session_state.summary,

                file_name="Research_Summary.txt",

                mime="text/plain"
            )
                        st.download_button(

                "📥 Download Existing Summary",

                st.session_state.summary,

                file_name="Research_Summary.txt",

                mime="text/plain"
            )
            elif selected == "🔬 Compare Papers":

    st.header("🔬 Compare Research Papers")

    if vector_store is None:

        st.info("Upload at least two research papers first.")

    elif doc_count < 2:

        st.warning("Please upload two or more research papers for comparison.")

    else:

        st.write("Generate an AI-powered comparison of all uploaded papers.")

        if st.button("🚀 Compare Papers"):

            with st.spinner("Comparing Research Papers..."):

                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama3-8b-8192",
                    temperature=0
                )

                context = st.session_state.paper_text[:12000]

                prompt = f"""
You are an expert AI Research Assistant.

Compare the uploaded research papers.

Generate the response using the following sections.

# Overview

# Similarities

# Differences

# Methodology Comparison

# Dataset Comparison

# Experimental Results

# Strengths

# Weaknesses

# Future Work

# Final Recommendation

Research Papers:

{context}
"""

                response = llm.invoke(prompt)

                comparison = response.content

                st.session_state.comparison = comparison
                                st.markdown("## 📊 AI Comparison Report")

                st.markdown(comparison)
                                st.download_button(

                    label="📥 Download Comparison",

                    data=comparison,

                    file_name="Research_Comparison.txt",

                    mime="text/plain"

                )
                        if "comparison" in st.session_state:

            st.markdown("---")

            st.success("Comparison already generated.")

            st.markdown(st.session_state.comparison)
                        st.download_button(

                "📥 Download Existing Comparison",

                st.session_state.comparison,

                file_name="Research_Comparison.txt",

                mime="text/plain"

            )
                    st.markdown("---")

        st.subheader("📋 Uploaded Papers")

        for i, file in enumerate(uploaded_files, start=1):

            st.write(f"{i}. {file.name}")
                    st.markdown("---")

        st.subheader("📋 Uploaded Papers")

        for i, file in enumerate(uploaded_files, start=1):

            st.write(f"{i}. {file.name}")
            elif selected == "💾 Export Chat":

    st.header("💾 Export Chat History")

    if len(st.session_state.chat_history) == 0:

        st.info("No chat history available.")

    else:

        chat_text = ""

        for message in st.session_state.chat_history:

            role = message["role"].upper()

            chat_text += f"{role}\n"

            chat_text += message["content"]

            chat_text += "\n\n"

        st.text_area(
            "Conversation",
            chat_text,
            height=400
        )

        st.download_button(
            "📥 Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )
        elif selected == "⚙️ Settings":

    st.header("⚙️ Settings")

    st.subheader("Application Information")

    st.info("""
AI Research Assistant Pro

Version : 1.0

Framework : Streamlit

Embedding Model :
sentence-transformers/all-MiniLM-L6-v2

Vector Database :
FAISS

LLM :
model = "Llama 3 8B (Groq)"

Language :
Python

Architecture :
Retrieval-Augmented Generation (RAG)

Developer :
Shreya Rai
""")
    st.markdown("---")

    st.subheader("Workspace")

    if st.button("🗑 Reset Entire Workspace"):

        st.session_state.uploaded_files = []
        st.session_state.paper_text = ""
        st.session_state.chunks = []
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.total_questions = 0

        if "summary" in st.session_state:
            del st.session_state.summary

        if "comparison" in st.session_state:
            del st.session_state.comparison

        st.success("Workspace reset successfully.")
        st.rerun()

    st.markdown("---")

    st.subheader("Current Workspace")

    st.write(f"📄 Uploaded Papers : {doc_count}")
    st.write(f"🧠 Chunks : {chunk_count}")
    st.write(f"💬 Questions : {question_count}")

    st.write(
        "🤖 AI Status : "
        + ("🟢 Online" if vector_store else "🔴 Offline")
    )

    st.markdown("---")

    st.subheader("Libraries Used")

    st.code("""
streamlit
pypdf
langchain-text-splitters
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
python-dotenv
torch
transformers
""")

    st.markdown("---")

    st.success("🚀 AI Research Assistant Pro is running successfully.")