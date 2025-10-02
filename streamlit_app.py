import streamlit as st
from rag_using_hf_api import run_chain_pipeline
import base64

# Inject custom CSS for modern SaaS look
st.markdown("""
    <style>
    body {
        min-height: 100vh;
        background: #181c2f;
        position: relative;
        overflow-x: hidden;
    }
    /* Animated gradient background */
    body:before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -2;
        min-height: 100vh;
        width: 100vw;
        background: linear-gradient(270deg, #181c2f 0%, #2e2e5d 25%, #3a3a7c 50%, #1e3a5c 75%, #0f2027 100%);
        animation: gradientMove 12s ease-in-out infinite;
        background-size: 400% 400%;
    }
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    /* Grunge overlay using SVG texture */
    body:after {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        pointer-events: none;
        background: url('data:image/svg+xml;utf8,<svg width=\'100%\' height=\'100%\' xmlns=\'http://www.w3.org/2000/svg\'><filter id=\'noise\'><feTurbulence type=\'fractalNoise\' baseFrequency=\'0.8\' numOctaves=\'2\' stitchTiles=\'stitch\'/></filter><rect width=\'100%\' height=\'100%\' filter=\'url(%23noise)\' opacity=\'0.08\'/></svg>');
        opacity: 0.18;
        mix-blend-mode: multiply;
    }
    .main-card {
        background: rgba(24,28,47,0.92);
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(60,72,88,0.18);
        padding: 2.5rem 2rem 2rem 2rem;
        margin-top: 2rem;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1.5px solid #6366f1;
        background: #23263a;
        font-size: 1.1rem;
        color: #e0e7ff;
    }
    .stTextInput>div>label {
        font-weight: 600;
        color: #60a5fa;
    }
    /* Center any button inside the main card */
    .main-card button {
        display: block !important;
        margin-left: auto !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #60a5fa 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(99,102,241,0.18);
    }
    .answer-box {
        background: #23263a;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        font-size: 1.15rem;
        color: #bfc4d1;
        box-shadow: 0 2px 8px rgba(99,102,241,0.12);
    }
    .footer {
        text-align: center;
        color: #60a5fa;
        font-size: 0.95rem;
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    .stExpander>div>div {
        background: #23263a;
        border-radius: 8px;
        color: #bfc4d1;
    }
    .stCodeBlock, .stMarkdown, .stText, .stExpanderContent {
        color: #bfc4d1 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-top:2.5rem; margin-bottom:2.5rem;'>
    <img src='https://img.icons8.com/fluency/48/youtube-play.png' style='vertical-align:middle; margin-bottom:8px;'>
    <span style='font-size:2.2rem; font-weight:700; color:#6366f1;'>YouTube RAG Chatbot</span>
    <div style='font-size:1.1rem; color:#64748b; margin-top:0.5rem;'>Ask questions about any YouTube video. Multilingual support. Powered by RAG & HuggingFace.</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    url_or_id = st.text_input(
        "YouTube Video URL or ID",
        "https://www.youtube.com/watch?v=OpUEEr-F5dY",
        help="Paste a YouTube video link or just the video ID."
    )
    question = st.text_input(
        "Your Question",
        "What is the video about? Give key highlights and overall summary",
        help="Type your question about the video."
    )
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=600,
        step=50,
        help="How large each transcript chunk is for processing. Larger chunks give more context but may be slower or less precise."
    )
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=80,
        step=10,
        help="How much content overlaps between consecutive chunks. More overlap helps preserve context between chunks."
    )
    k = st.number_input(
        "Number of Retrieved Documents (k)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many transcript chunks are retrieved for context. Higher values provide more information but may include less relevant content."
    )

    # HuggingFace model options
    model_name = st.selectbox(
        "LLM Model Name",
        [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "google/flan-t5-large",
            "tiiuae/falcon-7b-instruct",
            "HuggingFaceH4/zephyr-7b-beta"
        ],
        index=0,
        help="Choose which HuggingFace AI language model will generate answers. Different models vary in speed, accuracy, and resource usage."
    )

    embedding_model = st.selectbox(
        "Embedding Model Name",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5"
        ],
        index=0,
        help="Select the HuggingFace model used to convert transcript text into vectors for similarity search. Affects retrieval quality and performance."
    )

    status_box = st.empty()
    logs = []


    def status_callback(msg):
        # Filter out redundant status messages
        if msg.strip() in ["Running chain pipeline...", "--- Chain Pipeline Answer ---"]:
            return
        logs.append(msg)
        status_box.markdown("<br>".join(logs), unsafe_allow_html=True)


    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            try:
                answer, _ = run_chain_pipeline(
                    url_or_id,
                    question,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                    model_name=model_name,
                    embedding_model=embedding_model,
                    status_callback=status_callback
                )
                st.markdown(f"<div class='answer-box'><b>Answer:</b><br>{answer}</div>", unsafe_allow_html=True)
                with st.expander("Show Details/Logs"):
                    st.code("\n".join(logs))
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("""
<div class='footer'>
    &copy; 2025 YouTube RAG Chatbot &mdash; Powered by LangChain, Streamlit, and HuggingFace<br>
    <a href='https://github.com/rishiigupta04' target='_blank' style='text-decoration:none; margin:0 16px; display:inline-flex; align-items:center;'>
        <img src='https://img.icons8.com/color/48/github--v1.png' width='28' style='vertical-align:middle;'/>
        <span style='margin-left:8px; color:#e0e7ff; font-weight:500; font-size:1.05rem;'>My GitHub</span>
    </a>
    <a href='https://www.linkedin.com/in/rishirajgupta04/' target='_blank' style='text-decoration:none; margin:0 16px; display:inline-flex; align-items:center;'>
        <img src='https://img.icons8.com/color/48/linkedin.png' width='28' style='vertical-align:middle;'/>
        <span style='margin-left:8px; color:#e0e7ff; font-weight:500; font-size:1.05rem;'>My LinkedIn</span>
    </a>
</div>
""", unsafe_allow_html=True)