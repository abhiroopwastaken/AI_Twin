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
    """Load documents from knowledge_base and synthetic_data directories manually."""
    documents = []
    
    dirs_to_scan = [KNOWLEDGE_BASE_DIR, SYNTHETIC_DATA_DIR]
    
    for directory in dirs_to_scan:
        if os.path.exists(directory):
            print(f"Scanning {directory}...")
            files = os.listdir(directory)
            for filename in files:
                if filename.endswith(".txt"):
                    path = os.path.join(directory, filename)
                    try:
                        loader = TextLoader(path, encoding='utf-8')
                        documents.extend(loader.load())
                        print(f"Loaded {filename}")
                    except Exception as e:
                         # Raise error so it shows in app.py
                         raise Exception(f"Failed to load {path}: {str(e)}")
                         
    print(f"Loaded {len(documents)} documents total.")
    return documents

def create_vector_db():
    print("Starting vector DB creation...")
    documents = load_documents()
    if not documents:
        raise Exception("No documents loaded! Check knowledge_base folder permissions or content.")
        
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
