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
    
    # FORCE REBUILD ON STARTUP for deployment consistency
    # This ensures the app always uses the latest text files in knowledge_base
    try:
        import setup_db
        print("Forcing vector DB rebuild...")
        setup_db.create_vector_db()
    except Exception as e:
        st.error(f"Failed to rebuild vector DB: {e}")
    
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

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # User requested openai/gpt-oss-20b (Reverting from unsupported Step-3.5-Flash)
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
            max_new_tokens=2048,
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
    template = """You are an AI Twin representing Abhiroop Agarwal.

    Your purpose is to act as an intelligent, professional digital persona that answers questions about Abhiroop’s background, skills, projects, interests, experience, and career direction.

    CORE RULES:
    - Only use the provided context from the knowledge base.
    - Do not invent facts that are not present in the context.
    - If information is missing, try to answer from the facts available.
    - Clearly distinguish real and synthetic experiences when relevant.
    - Maintain factual accuracy and avoid exaggeration.

    TONE & STYLE:
    - Professional, concise, and thoughtful.
    - Speak in third person unless the question directly asks for a first-person answer.
    - Sound like a knowledgeable portfolio assistant, not a casual chatbot.
    - Avoid overly generic AI language.

    RESPONSE STRUCTURE:
    - Start with a clear direct answer.
    - Then provide supporting details from the context.
    - Keep answers structured and easy to read.

    PERSONA CONTEXT:
    Abhiroop is a software engineer transitioning into product management and technology strategy through an MBA in Information Management. His background combines enterprise engineering, AI applications, FinTech exposure, and growing product thinking.

    GOAL:
    Help recruiters, professors, or collaborators quickly understand Abhiroop’s profile through grounded and intelligent responses.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Inject profile text directly into context
    def format_docs_with_profile(docs):
        profile_text = get_profile_summary()
        retrieved_text = "\n\n".join(doc.page_content for doc in docs)
        return f"--- PROFILE DATA ---\n{profile_text}\n\n--- RETRIEVED KNOWLEDGE ---\n{retrieved_text}"

    # LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs_with_profile, "question": RunnablePassthrough()}
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
    # Profile Image
    profile_image_path = os.path.join("media", "Passport_image.jpeg")
    if os.path.exists(profile_image_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(profile_image_path, width='stretch')
        
    st.header("Profile Summary")
    st.text_area("About Me", value=get_profile_summary(), height=400, disabled=True)
    st.markdown("---")
    st.info("This AI allows you to ask questions about my professional background, skills, and projects.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Ask Me Anything", "Recruiter View", "Download CV", "Express Interest"])

with tab1:
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def handle_user_input(user_question):
        """Handle user input and generate response."""
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Generate response
        rag_chain = setup_rag_chain()
        if rag_chain:
            try:
                response = rag_chain.invoke(user_question)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error("An error occurred.")
                st.code(traceback.format_exc())
        else:
                st.error("RAG system not initialized.")
        # Rerun to update display
        st.rerun()

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sample Questions (Always visible)
    st.markdown("###### Suggested Questions:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Based on Abhiroop’s background, what roles is he best suited for and why?", width='stretch'):
            handle_user_input("Based on Abhiroop’s background, what roles is he best suited for and why?")
        if st.button("How has Abhiroop applied AI or generative AI in his work?", width='stretch'):
            handle_user_input("How has Abhiroop applied AI or generative AI in his work?")
        if st.button("Summarize Abhiroop’s professional journey and long-term vision.", width='stretch'):
            handle_user_input("Summarize Abhiroop’s professional journey and long-term vision.")

    with col2:
        if st.button("What differentiates Abhiroop from other software engineers?", width='stretch'):
           handle_user_input("What differentiates Abhiroop from other software engineers?")
        if st.button("What leadership experiences demonstrate Abhiroop’s growth potential?", width='stretch'):
            handle_user_input("What leadership experiences demonstrate Abhiroop’s growth potential?")

    # Handle Input
    if prompt := st.chat_input("Ask me about my experience, skills, or projects..."):
        handle_user_input(prompt)

    # JS to scroll to bottom (Only for Tab 1)
    st.markdown(
        """
        <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTop = body.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True
    )

with tab2:
    st.header("Recruiter View")
    st.markdown("**AI-Generated Recruiter Brief**")
    st.markdown("This tab generates a professional, AI-powered recruiter summary of Abhiroop — covering his skills, fit, and standout qualities. Click **Generate** to create it fresh.")
    
    if st.button("⚡ Generate Recruiter Brief"):
        with st.spinner("Analyzing profile and generating brief..."):
            rag_chain = setup_rag_chain()
            if rag_chain:
                try:
                    # Specific prompt for recruiter brief
                    recruiter_prompt = """
                    Act as an expert technical recruiter. Based on the context provided about Abhiroop Agarwal, generate a "Recruiter Brief".
                    
                    Use the provided context to find his Location, Education, and Key Role.
                    
                    Format the response exactly as follows using Markdown:
                    
                    ### Recruiter Brief: Abhiroop Agarwal
                    
                    **Professional Snapshot**: [A compelling 3-4 sentence summary of his background, experience, and unique value proposition.]
                    
                    **Top 3 Hard Skills with Evidence**:
                    1. **[Skill 1]**: [Brief evidence/project]
                    2. **[Skill 2]**: [Brief evidence/project]
                    3. **[Skill 3]**: [Brief evidence/project]
                    
                    **Top 3 Soft Skills with Evidence**:
                    1. **[Skill 1]**: [Brief evidence/context]
                    2. **[Skill 2]**: [Brief evidence/context]
                    3. **[Skill 3]**: [Brief evidence/context]
                    
                    **Quick Stats**:
                    | Metric | Value |
                    | :--- | :--- |
                    | **Education** | [Degree/Institute] |
                    | **Key Role** | [Most relevant past role] |
                    | **Focus Area** | [Product/Tech/Strategy] |
                    | **Location** | [City from profile] |
                    """
                    
                    response = rag_chain.invoke(recruiter_prompt)
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating brief: {str(e)}")
            else:
                st.error("RAG system not initialized.")

with tab3:
    st.header("Download CV")
    resume_path = os.path.join("media", "Abhiroop_Agarwal_Resume.pdf")
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            pdf_data = f.read()
        st.download_button(
            label="📄 Download Resume (PDF)",
            data=pdf_data,
            file_name="Abhiroop_Agarwal_Resume.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("Resume file not found. Please add 'Abhiroop_Agarwal_Resume.pdf' to the 'media' directory.")

with tab4:
    st.header("Express Interest")
    st.write("Interested in collaborating? Fill out the form below!")
    
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Email Address")
        role = st.selectbox("Role / Opportunity", ["Recruiter", "Freelance Project", "Collaboration", "Other"])
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send Message")
        
        if submitted:
            if name and email and message:
                st.success("Thanks for reaching out! I'll get back to you soon.")
            else:
                st.error("Please fill in all required fields.")
