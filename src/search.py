import os
from dotenv import load_dotenv 
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class RAGSearch:
    """Manages RAG search and summarization using FAISS and Gemini."""

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemini-3-flash-preview"
    ):
        """
        Initialize RAG Search.

        Args:
            persist_dir: Directory for FAISS vector storage.
            embedding_model: Hugging Face embedding model.
            llm_model: Google Gemini model name.
        """

        # Initialize FAISS vector store
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Check whether FAISS index and metadata exist
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        # Load existing vector store or build a new one
        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):
            print("[INFO] Building new FAISS vector store...")

            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            if not docs:
                raise ValueError("No documents found in the data directory.")

            self.vectorstore.build_from_documents(docs)

            print("[INFO] FAISS vector store built successfully.")

        else:
            print("[INFO] Loading existing FAISS vector store...")
            self.vectorstore.load()

        # Get Google API key from .env
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Check your .env file."
            )

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=1024
        )

        print(f"[INFO] Gemini LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Search relevant documents and generate a summary using Gemini.

        Args:
            query: User's question.
            top_k: Number of relevant documents to retrieve.

        Returns:
            Gemini-generated answer.
        """

        # 1. Retrieve relevant documents
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # 2. Extract text from retrieved metadata
        texts = [
            r.get("metadata", {}).get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(
            text for text in texts if text
        )

        if not context:
            return "No relevant documents found."

        # 3. Create prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the following retrieved context.

Instructions:
- Use the provided context to answer.
- Be accurate and concise.
- Do not invent information.
- If the context does not contain the answer, say so.

User Question:
{query}

Retrieved Context:
{context}

Answer:
"""

        # 4. Generate response using Gemini
        response = self.llm.invoke(prompt)

        return response.text