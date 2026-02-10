import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Constants
KNOWLEDGE_BASE_DIR = "knowledge_base"
SYNTHETIC_DATA_DIR = "synthetic_data"
VECTOR_DB_PATH = "vector_db"

def load_documents():
    """Load documents from knowledge_base and synthetic_data directories."""
    documents = []
    
    # Load from knowledge_base
    if os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Loading documents from {KNOWLEDGE_BASE_DIR}...")
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="*.txt", loader_cls=TextLoader)
        documents.extend(loader.load())
        
    # Load from synthetic_data
    if os.path.exists(SYNTHETIC_DATA_DIR):
        print(f"Loading documents from {SYNTHETIC_DATA_DIR}...")
        loader = DirectoryLoader(SYNTHETIC_DATA_DIR, glob="*.txt", loader_cls=TextLoader)
        documents.extend(loader.load())
        
    print(f"Loaded {len(documents)} documents.")
    return documents

def create_vector_db():
    print("Starting vector DB creation...")
    documents = load_documents()
    if not documents:
        print("No documents found. Please check data directories.")
        return
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    
    print("Creating embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building vector store...")
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"Vector store created at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    create_vector_db()
