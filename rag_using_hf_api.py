from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from warnings import filterwarnings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os

filterwarnings("ignore", category=UserWarning, module="langchain_community.vectorstores.faiss", lineno=0)

# Load environment variables (HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, FEATHERLESS_API_KEY should be in .env file)
load_dotenv()

# Check for HuggingFace and Featherless API tokens
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
featherless_token = os.getenv("FEATHERLESS_API_KEY")

if not hf_token and not featherless_token:
    raise EnvironmentError(
        "Neither HuggingFace nor Featherless API token found. Please create a .env file in your project directory with the line HF_TOKEN=your_huggingface_api_token_here or HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here, and/or FEATHERLESS_API_KEY=your_featherless_api_key_here, or set the environment variables."
    )


# =====================
# 1. Transcript Handling
# =====================

def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the YouTube video ID from a URL or returns the ID if already provided.
    Supports common YouTube URL formats.
    """
    if re.match(r'^[\w-]{11}$', url_or_id):
        return url_or_id
    match = re.search(r'(?:v=|\/embed\/|youtu\.be\/|\/v\/|\/shorts\/)([\w-]{11})', url_or_id)
    if match:
        return match.group(1)
    match = re.search(r'([\w-]{11})', url_or_id)
    if match:
        return match.group(1)
    raise ValueError("Could not extract a valid YouTube video ID from input.")


transcript_cache = {}


def fetch_transcript(video_id: str, language: str = "en") -> list:
    """
    Fetch transcript from a YouTube video using its ID.
    Returns a list of dicts: [{"text": ..., "start": ...}, ...]
    """
    cache_key = f"{video_id}:{language}"
    if cache_key in transcript_cache:
        return transcript_cache[cache_key]
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id, languages=[language])
        transcript = [{"text": snippet.text, "start": snippet.start} for snippet in transcript_data]
        transcript_cache[cache_key] = transcript
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video.")
        transcript_cache[cache_key] = []
        return []


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Returns the language code (e.g., 'en', 'hi').
    """
    try:
        # Get full text from transcript segments
        if isinstance(text, list):
            text = " ".join([seg.get("text", "") for seg in text[:10]])
        return detect(text)
    except Exception:
        return "en"  # Default to English if detection fails


def translate_to_english(transcript: list, src_lang: str) -> list:
    """
    Translate transcript segments to English using GoogleTranslator if not already English.
    """
    if src_lang == "en":
        return transcript
    try:
        full_text = " ".join([seg["text"] for seg in transcript])
        translated = GoogleTranslator(source=src_lang, target="en").translate(full_text)
        print(f"Transcript translated from {src_lang} to English.")
        # Return as a single segment with start time 0
        return [{"text": translated, "start": 0}]
    except Exception as e:
        print(f"Translation failed: {e}")
        return transcript


# =====================
# 2. Chunking and Text Processing
# =====================

def split_text_with_splitter(transcript_segments, chunk_size: int = 600, chunk_overlap: int = 80):
    """
    Use RecursiveCharacterTextSplitter to split transcript segments into chunks, preserving timestamps.
    Returns a list of chunk dicts: [{"text": ..., "start": ...}]
    """
    # Combine all transcript segments into a single string, but keep track of start positions
    texts = [seg["text"] for seg in transcript_segments]
    starts = [seg["start"] for seg in transcript_segments]
    joined_text = "\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""],
    )
    chunks = splitter.split_text(joined_text)
    # Map each chunk to the closest original start timestamp
    chunk_dicts = []
    current_pos = 0
    for chunk in chunks:
        # Find the start index in the joined_text
        idx = joined_text.find(chunk, current_pos)
        if idx == -1:
            idx = current_pos  # fallback
        # Find which original segment this position belongs to
        char_count = 0
        start_time = 0
        for text, start in zip(texts, starts):
            char_count += len(text) + 1  # +1 for the newline
            if char_count > idx:
                start_time = start
                break
        chunk_dicts.append({"text": chunk.strip(), "start": start_time})
        current_pos = idx + len(chunk)
    return chunk_dicts


# =====================
# 3. Embedding & Retrieval
# =====================

from langchain_core.documents import Document


def create_vector_store(chunks, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Create a FAISS vector store from text chunks using HuggingFace embeddings.
    Each chunk is converted to a Document with timestamp metadata.
    Returns the vector store object.
    """
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    docs = [Document(page_content=chunk["text"], metadata={"start": chunk["start"]}) for chunk in chunks]
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
    return vector_store


def get_retriever(vector_store, k: int = 5):
    """
    Create a retriever to find relevant chunks for a question.
    Returns the retriever object.
    """
    retrieved_docs = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retrieved_docs


def format_docs(retrieved_docs):
    """
    Combine the text from multiple retrieved documents into a single string, including start-end timestamps in HH:MM:SS format.
    If the estimated end time equals the start time, only show the start time.
    The end time is estimated by adding a duration based on chunk length, but capped at the last transcript segment's start time.
    """

    def fmt_ts(ts):
        if ts is None:
            return "??:??:??"
        ts = int(ts)
        h = ts // 3600
        m = (ts % 3600) // 60
        s = ts % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # Find the maximum start time (last transcript segment)
    max_start = max([doc.metadata.get("start", 0) for doc in retrieved_docs if hasattr(doc, "metadata")], default=0)
    formatted = []
    for doc in retrieved_docs:
        start = doc.metadata.get("start") if hasattr(doc, "metadata") else None
        est_duration = max(5, len(doc.page_content) // 10)
        end = start + est_duration if start is not None else None
        # Cap end time at last transcript segment
        if end is not None and end > max_start:
            end = max_start
        # If end == start, only show start time
        if start is not None and end is not None and int(start) == int(end):
            ts_range = f"[Timestamp: {fmt_ts(start)}]"
        elif start is not None and end is not None:
            ts_range = f"[Timestamp: {fmt_ts(start)} - {fmt_ts(end)}]"
        else:
            ts_range = ""
        formatted.append(f"{ts_range} {doc.page_content}")
    return "\n\n".join(formatted)


# =====================
# 4. Prompt & LLM
# =====================

def build_prompt():
    """
    Create a prompt template for the language model, instructing it to cite timestamps for each point and provide a summary of all source timestamps at the end.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant that helps people find information.\n"
            "Use the following pieces of context to answer the question at the end.\n"
            "For each point in your answer, cite the source timestamp from the context, mentioning it before the point.\n"
            "At the end of your answer, provide a summary list of all source timestamps used. Give each of them a heading based on the context to what the timestamps refer to, followed by the actual timestamp for that particular header\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
            "{context}\n"
            "Question: {question}"
        ),
    )


def get_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", temperature: float = 0.2):
    """
    Get the language model for answering questions using HuggingFace or Featherless API.
    Automatically selects the correct API key based on the model name.
    """
    # Detect provider by model name (simple heuristic, adjust as needed)
    if "featherless" in model_name.lower():
        if not featherless_token:
            raise EnvironmentError("Featherless API token (FEATHERLESS_API_KEY) not found. Please add it to your .env file.")
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=512,
            huggingfacehub_api_token=featherless_token
        )
    else:
        if not hf_token:
            raise EnvironmentError("HuggingFace API token (HF_TOKEN or HUGGINGFACEHUB_API_TOKEN) not found. Please add it to your .env file.")
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=512,
            huggingfacehub_api_token=hf_token
        )
    model = ChatHuggingFace(llm=llm)
    return model


# =====================
# 5. Main Pipeline
# =====================

# Simple in-memory per-user rate limiting: 3 queries per user per session (not persistent)
MAX_QUERIES_PER_USER = 3
user_query_counts = {}

def run_chain_pipeline(url_or_id, question, user_id='default_user', chunk_size=600, chunk_overlap=80, k=3,
                       model_name="mistralai/Mistral-7B-Instruct-v0.2",
                       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                       status_callback=None):
    count = user_query_counts.get(user_id, 0)
    if count >= MAX_QUERIES_PER_USER:
        msg = f"User '{user_id}' has reached the session query limit of {MAX_QUERIES_PER_USER}. Restart the script to reset."
        print(msg)
        return msg, msg
    user_query_counts[user_id] = count + 1
    def status(msg):
        print(msg)
        if status_callback:
            status_callback(msg)
    status("Getting transcript...")
    transcript = fetch_transcript(extract_video_id(url_or_id))
    if not transcript:
        status("Transcript not found.")
        return "Transcript not found.", "Transcript not found."
    status("Detecting transcript language...")
    lang = detect_language(transcript)
    status(f"Transcript language detected: {lang}")
    transcript = translate_to_english(transcript, lang)
    status("Splitting transcript into chunks...")
    chunks = split_text_with_splitter(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    status(f"Number of chunks: {len(chunks)}")
    status("Creating vector store...")
    vector_store = create_vector_store(chunks, embedding_model=embedding_model)
    status("Setting up retriever...")
    retriever = get_retriever(vector_store, k=k)
    status("Building prompt and LLM...")
    prompt = build_prompt()
    model = get_llm(model_name=model_name)
    parser = StrOutputParser()
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    sequential_chain = prompt | model | parser
    final_chain = parallel_chain | sequential_chain

    status("Running chain pipeline...")
    result = final_chain.invoke(question)
    print("\n--- Chain Pipeline Answer ---")
    print(result)
    return str(result), None


# =====================
# 6. Entry Point
# =====================

def main():
    url_or_id = "https://www.youtube.com/watch?v=OpUEEr-F5dY"
    question = "Who is the major suspect? What all do we know of them?"
    run_chain_pipeline(url_or_id, question)


if __name__ == "__main__":
    main()