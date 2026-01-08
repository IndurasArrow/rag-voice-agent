import os
import tempfile
import uuid
import asyncio
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from gtts import gTTS

load_dotenv()

def init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        "google_api_key": "",
        "setup_complete": False,
        "vector_store": None,
        "processed_documents": [],
        "messages": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar() -> None:
    """Configure sidebar with API settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        st.session_state.google_api_key = st.text_input(
            "Google API Key",
            value=st.session_state.google_api_key,
            type="password",
            help="Enter your Google Gemini API Key"
        )
        
        if st.session_state.processed_documents:
            st.markdown("---")
            st.header("📚 Knowledge Base")
            for doc in st.session_state.processed_documents:
                st.caption(f"📄 {doc}")
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This agent uses **Google Gemini 1.5** for reasoning and **FAISS** for retrieval. "
            "Upload a PDF, ask a question, and get a voice response!"
        )

def process_pdf(file) -> List:
    """Process PDF file and split into chunks with metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name
                })
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []

def update_vector_store(documents: List) -> None:
    """Update or create FAISS vector store."""
    if not st.session_state.google_api_key:
        raise ValueError("Google API Key not provided")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.session_state.google_api_key
    )
    
    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
    else:
        st.session_state.vector_store.add_documents(documents)

def generate_gemini_response(context: str, query: str, api_key: str) -> str:
    """Generate response using Google Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.    
Context:
{context}

User Question: {query}

Please provide a clear, concise answer that can be easily spoken out loud. Avoid using markdown formatting like bold or italics in the response text as it will be read by text-to-speech."""

    response = model.generate_content(prompt)
    return response.text

def generate_audio(text: str) -> str:
    """Generate MP3 using gTTS."""
    tts = gTTS(text=text, lang='en', slow=False)
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
    tts.save(audio_path)
    return audio_path

async def process_query(query: str) -> Dict:
    """Process user query and generate voice response."""
    try:
        if not st.session_state.vector_store:
            raise Exception("Vector store not initialized")

        # Search documents
        search_results = st.session_state.vector_store.similarity_search(query, k=3)
        
        if not search_results:
            raise Exception("No relevant documents found")
            
        # Prepare context
        context = ""
        sources = []
        for i, doc in enumerate(search_results, 1):
            source = doc.metadata.get('file_name', 'Unknown Source')
            context += f"From {source}:\n{doc.page_content}\n\n"
            sources.append(source)
            
        # Generate text response
        text_response = generate_gemini_response(
            context, 
            query, 
            st.session_state.google_api_key
        )
        
        # Generate audio
        audio_path = generate_audio(text_response)
        
        return {
            "status": "success",
            "text_response": text_response,
            "audio_path": audio_path,
            "sources": list(set(sources))
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def main() -> None:
    """Main application function."""
    st.set_page_config(
        page_title="Voice RAG Agent",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stChatInput {
            padding-bottom: 20px;
        }
        .stSpinner {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    setup_sidebar()
    
    st.title("🎙️ Voice RAG Agent")
    st.caption("Powered by **Gemini 2.5 Flash** & **FAISS**")
    
    # File upload section in main area if no docs yet
    if not st.session_state.setup_complete:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; border: 2px dashed #cccccc; border-radius: 10px; margin-bottom: 2rem;'>
            <h3>👋 Welcome!</h3>
            <p>To get started, please enter your Google API Key in the sidebar and upload a PDF document.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
            
            if uploaded_file:
                if not st.session_state.google_api_key:
                    st.warning("⚠️ Please provide your Google API Key in the sidebar first!")
                else:
                    file_name = uploaded_file.name
                    if file_name not in st.session_state.processed_documents:
                        with st.status("🚀 Processing document...", expanded=True) as status:
                            try:
                                st.write("📄 Parsing PDF content...")
                                documents = process_pdf(uploaded_file)
                                if documents:
                                    st.write("🧠 Generating embeddings & indexing...")
                                    update_vector_store(documents)
                                    st.session_state.processed_documents.append(file_name)
                                    st.session_state.setup_complete = True
                                    status.update(label="✅ Ready to chat!", state="complete", expanded=False)
                                    st.rerun()
                            except Exception as e:
                                status.update(label="❌ Error processing", state="error")
                                st.error(f"Error: {str(e)}")

    else:
        # Chat Interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "audio_path" in message:
                    st.audio(message["audio_path"], format="audio/mp3")
                    with open(message["audio_path"], "rb") as f:
                         st.download_button("⬇️ Download Audio", f, file_name="response.mp3", mime="audio/mp3")
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.setup_complete:
                 st.error("Please upload a document first.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = asyncio.run(process_query(prompt))
                    
                    if result["status"] == "success":
                        st.markdown(result["text_response"])
                        
                        # Audio and controls
                        col_audio, col_dl = st.columns([3, 1])
                        with col_audio:
                            st.audio(result["audio_path"], format="audio/mp3")
                        with col_dl:
                            with open(result["audio_path"], "rb") as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f,
                                    file_name="response.mp3",
                                    mime="audio/mp3"
                                )
                        
                        # Sources
                        with st.expander("View Sources"):
                            for source in result["sources"]:
                                st.write(f"- {source}")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["text_response"],
                            "audio_path": result["audio_path"],
                            "sources": result["sources"]
                        })
                    else:
                        st.error(f"Error: {result.get('error')}")

if __name__ == "__main__":
    main()