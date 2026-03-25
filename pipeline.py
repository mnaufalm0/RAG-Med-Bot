import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

STORE_PATH  = Path("vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODELS = {
    "fast":    "llama-3.1-8b-instant",
    "default": "llama-3.3-70b-versatile",
    "mixtral": "mixtral-saba-24b",
}

PROMPT_TEMPLATE = """You are MediRAG, a health assistant backed by a RAG pipeline over a disease dataset.
Answer only from the context below. If the context doesn't cover the question, say so and suggest seeing a doctor.

Context:
{context}

Guidelines:
- Use clear sections: Overview, Symptoms, Causes, Treatment, Precautions
- Cite which disease the info comes from
- Keep it readable — bullet points where it helps
- End every response with a disclaimer to consult a real doctor

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


@dataclass
class Result:
    answer: str
    source_docs: list[Document]
    diseases: list[str]

    def sources(self) -> list[str]:
        seen = set()
        out = []
        for doc in self.source_docs:
            label = f"{doc.metadata.get('disease', '?')} · {doc.metadata.get('source', '?')}"
            if label not in seen:
                seen.add(label)
                out.append(label)
        return out


class MediRAG:
    def __init__(self, groq_api_key: Optional[str] = None,model: str = "default",k: int = 5,temperature: float = 0.1,verbose: bool = False,):
        self.k = k
        self.verbose = verbose
        self.model_name = MODELS.get(model, MODELS["default"])

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

        if verbose:
            print(f"loading embedder: {EMBED_MODEL}")
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not STORE_PATH.exists():
            raise FileNotFoundError(
                f"No vector store found at '{STORE_PATH}'. Run: python build_index.py"
            )
        if verbose:
            print(f"loading FAISS index from {STORE_PATH}")
        self.store = FAISS.load_local(
            str(STORE_PATH),
            self.embedder,
            allow_dangerous_deserialization=True,
        )

        if verbose:
            print(f"connecting to Groq: {self.model_name}")
        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=api_key,
            temperature=temperature,
            max_tokens=1024,
        )

        
        self.retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3},
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

    def ask(self, question: str) -> Result:
        if self.verbose:
            print(f"query: {question}")

        result = self.chain.invoke({"query": question})
        docs = result.get("source_documents", [])
        diseases = sorted({d.metadata.get("disease", "") for d in docs if d.metadata.get("disease")})

        return Result(
            answer=result["result"],
            source_docs=docs,
            diseases=diseases,
        )

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Raw similarity search — handy for debugging."""
        return self.store.similarity_search(query, k=k)

    @property
    def meta(self) -> dict:
        path = STORE_PATH / "metadata.json"
        return json.loads(path.read_text()) if path.exists() else {}
