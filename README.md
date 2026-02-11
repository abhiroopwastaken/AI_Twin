# AI Twin - Abhiroop Agarwal 

Your intelligent, professional digital persona. This AI Twin is designed to represent **Abhiroop Agarwal**, answering questions about his background, skills, projects, and career aspirations, interacting just like a knowledgeable portfolio assistant.

##  Features

*   **Interactive Chat**: Ask anything about Abhiroop's professional journey. The AI responds using a Retrieval-Augmented Generation (RAG) pipeline grounded in his actual profile data.
*   **Recruiter View**: One-click generation of a "Recruiter Brief" — a concise professional summary highlighting hard skills, soft skills, and key stats.
*   **Resume Download**: Easy access to the latest PDF resume.
*   **Express Interest**: A built-in contact form for recruiters and collaborators to reach out directly.
*   **Admin Tools**: Server-side controls to manual refresh the knowledge base (Vector DB) ensuring the AI is always up-to-date.

##  Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (Python-based UI).
*   **LLM Integration**: [LangChain](https://www.langchain.com/) + Hugging Face Inference API.
*   **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search).
*   **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers).
*   **Model**: Open-source models (via Hugging Face).

##  Installation & Setup

### Prerequisites
*   Python 3.9+
*   A Hugging Face API Token

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/abhiroopwastaken/AI_Twin.git
    cd AI_Twin
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```ini
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
    ```

5.  **Build the Knowledge Base**
    Run the setup script to ingest documents from `knowledge_base/` and build the vector database:
    ```bash
    python setup_db.py
    ```

6.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

*   `app.py`: Main application logic (UI & RAG pipeline).
*   `setup_db.py`: Script to generate embeddings and build the FAISS index.
*   `knowledge_base/`: Text files containing raw profile data.
*   `vector_db/`: Generated FAISS index files (the "Brain").
*   `media/`: Images and static assets (Resume, Profile Pic).

##  License

[MIT](LICENSE)
