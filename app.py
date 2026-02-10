import os
import streamlit as st
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Constants
KNOWLEDGE_BASE_DIR = "knowledge_base"
SYNTHETIC_DATA_DIR = "synthetic_data"
VECTOR_DB_PATH = "vector_db"

# Set page config
st.set_page_config(page_title="Abhiroop's AI Twin", page_icon="🧠", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    h1 {
        color: #f0f2f6;
        font-weight: 600;
    }
    
    .stChatMessage {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #2b313e;
        border-left: 5px solid #4a90e2;
    }
    
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1c1e24;
        border-left: 5px solid #ff4b4b;
    }
    
    .stSidebar {
        background-color: #161a23;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vector_db():
    """Load the existing vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def setup_rag_chain():
    """Initialize the RAG chain using LCEL."""
    vector_db = load_vector_db()
    
    if not vector_db:
        # Auto-build vector DB if missing (for deployment)
        with st.spinner("Building vector database... this may take a moment"):
             try:
                 import setup_db
                 setup_db.create_vector_db()
                 
                 # IMPORTANT: Clear cache to force reload of the new DB
                 load_vector_db.clear()
                 vector_db = load_vector_db()
             except Exception as e:
                 st.error(f"Error building vector DB: {str(e)}")
                 st.code(traceback.format_exc())
             
    if not vector_db:
        st.error("Failed to load or create vector database.")
        st.info("Debugging: Ensure 'knowledge_base' folder exists and contains .txt files.")
        return None

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    
    # User requested openai/gpt-oss-20b
    repo_id = "openai/gpt-oss-20b"
    # Try to get token from environment or streamlit secrets
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token and "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    
    if not api_token:
        st.error("Missing Hugging Face Token.")
        st.info("""
        **To fix this on Streamlit Cloud:**
        1. Go to your App Dashboard
        2. Click 'Manage app' -> 'Settings' -> 'Secrets'
        3. Add the following:
           ```toml
           HUGGINGFACEHUB_API_TOKEN = "hf_..."
           ```
        4. Reboot the app.
        """)
        st.stop()

    try:
        # Initialize Endpoint
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=api_token
        )
        # Wrap in ChatHuggingFace for better compatibility with instruction tuned models
        chat_model = ChatHuggingFace(llm=llm)
        
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

    # Create Prompt Template
    template = """You are an AI Twin representation of a professional. 
    Use the following pieces of context to answer the question at the end.
    If the answer is not in the context, say you don't know, but try to infer from the profile if possible.
    Keep answers professional and concise.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    
    return rag_chain

def get_profile_summary():
    """Read profile.txt for summary display."""
    try:
        with open(os.path.join(KNOWLEDGE_BASE_DIR, "profile.txt"), "r") as f:
            return f.read()
    except:
        return "Profile information not available."

# Main App UI
# Main App UI
st.title("Abhiroop Agarwal's AI Twin")
st.markdown("ask me anything about my professional background, skills, and projects.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Profile Summary")
    st.text_area("About Me", value=get_profile_summary(), height=400, disabled=True)
    st.markdown("---")
    st.info("This AI allows you to ask questions about my professional background, skills, and projects.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about my experience, skills, or projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_chain = setup_rag_chain()
            if rag_chain:
                try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("An error occurred.")
                    st.code(traceback.format_exc())
            else:
                 st.error("RAG system not initialized.")
