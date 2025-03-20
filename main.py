import os
import streamlit as st
import torch
import torchaudio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import whisper

# ✅ Fix for ChromaDB sqlite3 issue
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# ✅ Load Whisper model
def load_whisper_model():
    """Loads the Whisper model for speech-to-text conversion."""
    model = whisper.load_model("base")
    return model

# ✅ Convert audio to text

def convert_audio_to_text(audio_file, model):
    """Converts input audio to text using Whisper."""
    waveform, sample_rate = torchaudio.load(audio_file)
    audio = waveform.squeeze().numpy()
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

# ✅ Initialize Gemini LLM
def initialize_llm():
    """Initializes the Gemini LLM using an API key."""
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCDhtBmRgD88X1VX8TTF30C9Iixc2fVpw0")

# ✅ Load subtitle dataset and apply chunking (Improved chunk overlap)
def load_and_process_subtitles(directory="dataset"):
    """Loads subtitle files from a directory and processes them into a retriever."""
    loader = DirectoryLoader(directory, glob="*_cleaned.srt", loader_cls=TextLoader)
    documents = loader.load()
    
    # 🔹 **Improved Chunking**: Overlapping ensures better context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 🔹 Use Sentence-BERT Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 🔹 Store embeddings in ChromaDB
    db = Chroma.from_documents(texts, embeddings)
    return db

# ✅ Query LLM for additional context
def query_llm(llm, query):
    """Queries the LLM for context and retrieves relevant information."""
    response = llm.invoke(query)
    return response

# ✅ Main Streamlit App
def main():
    st.title("Subtitle Retriever")
    st.write("Upload a **TV/Movie audio clip** to find matching subtitles.")

    whisper_model = load_whisper_model()
    llm = initialize_llm()
    retriever = load_and_process_subtitles()

    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        with st.spinner("Transcribing audio..."):
            transcription = convert_audio_to_text(uploaded_file, whisper_model)
        st.text_area("Transcription:", transcription, height=150)

        with st.spinner("Searching subtitles..."):
            docs = retriever.similarity_search(transcription, k=3)  # 🔹 Returns 3 most relevant results

        # ✅ Display retrieved subtitles and context
        st.write("### Retrieved Subtitles and Context")
        for doc in docs:
            movie_name = doc.metadata.get("source", "Unknown Movie")
            subtitle_text = doc.page_content

            llm_query = f"Extract key details from this subtitle: {subtitle_text}"
            context = query_llm(llm, llm_query)

            st.subheader(movie_name)
            st.write(f"**Subtitle:** {subtitle_text}")
            st.write(f"**Context:** {context}")

if __name__ == "__main__":
    main()
