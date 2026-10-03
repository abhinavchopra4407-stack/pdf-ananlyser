import os
import sys
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from services.embeddings import EmbeddingService

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from groq import Groq
except ImportError:
    Groq = None


RAG_PROMPT_TEMPLATE = """You are a precise, helpful academic teaching assistant.
Answer the user's question ONLY using the provided PDF document context excerpts below.

CRITICAL INSTRUCTIONS:
1. Base your answer STRICTLY on the provided context. Do NOT invent facts or concepts not supported by the context.
2. IF THE ANSWER CANNOT BE FOUND OR INFERRED FROM THE CONTEXT, state clearly:
   "I could not find the answer to this question in the uploaded PDF document."
3. MANDATORY: ALWAYS cite the specific PDF Page numbers where the information came from (e.g., [Page 12] or [Pages 14, 18]).
4. Maintain a clear, educational, and easy-to-understand tone.

Context Excerpts from PDF:
{context_str}

User Question: {question}
"""


class RAGPipeline:
    """RAG Service for chunking, vector indexing in ChromaDB, semantic retrieval, and QA with page citations."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_MODEL
        self.embedding_service = EmbeddingService()
        self.chroma_client = None
        self.collection = None
        self.doc_id = None
        self.chat_history: List[Dict[str, str]] = []
        self._init_chroma()
        self._init_groq()

    def _init_chroma(self):
        if chromadb is not None:
            try:
                persist_path = str(config.CHROMA_DIR)
                self.chroma_client = chromadb.PersistentClient(path=persist_path)
            except Exception as e:
                print(f"Warning: ChromaDB initialization issue ({e}). Using in-memory store.")
                self.chroma_client = chromadb.Client()

    def _init_groq(self):
        if self.api_key and Groq is not None:
            try:
                self.groq_client = Groq(api_key=self.api_key)
            except Exception:
                self.groq_client = None
        else:
            self.groq_client = None

    def update_credentials(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        if model_name:
            self.model_name = model_name
        self._init_groq()

    def clear_session(self):
        """Resets the vector database collection and chat history for a new document upload."""
        self.chat_history.clear()
        if self.chroma_client and self.doc_id:
            try:
                self.chroma_client.delete_collection(name=self.doc_id)
            except Exception:
                pass
        self.doc_id = None
        self.collection = None

    def create_chunks(self, pages_data: List[Dict[str, Any]], chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Splits page text into smaller chunks while strictly preserving original PDF page number metadata.
        """
        chunks = []
        chunk_id = 0

        for page in pages_data:
            page_num = page["page_number"]
            text = page["text"]
            if not text or len(text.strip()) < 10:
                continue

            # Split text into sentences / paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]

            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) <= chunk_size:
                    current_chunk += ("\n\n" if current_chunk else "") + p
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": f"p{page_num}_c{chunk_id}",
                            "text": current_chunk,
                            "page_number": page_num
                        })
                        chunk_id += 1
                        # Maintain overlap
                        overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                        current_chunk = overlap_text + "\n\n" + p
                    else:
                        # Paragraph itself is larger than chunk size
                        chunks.append({
                            "chunk_id": f"p{page_num}_c{chunk_id}",
                            "text": p[:chunk_size],
                            "page_number": page_num
                        })
                        chunk_id += 1
                        current_chunk = ""

            if current_chunk:
                chunks.append({
                    "chunk_id": f"p{page_num}_c{chunk_id}",
                    "text": current_chunk,
                    "page_number": page_num
                })
                chunk_id += 1

        return chunks

    def build_vector_index(self, pages_data: List[Dict[str, Any]], filename: str) -> int:
        """
        Chunks the document, computes embeddings, and indexes them in ChromaDB.
        Returns the total number of chunks indexed.
        """
        self.clear_session()

        # Sanitize collection name for ChromaDB (3-63 chars, alphanumeric)
        clean_name = "".join(c for c in Path(filename).stem if c.isalnum()).lower()
        if len(clean_name) < 3:
            clean_name = "doc" + clean_name
        self.doc_id = f"pdf_{clean_name[:25]}_{uuid.uuid4().hex[:6]}"

        chunks = self.create_chunks(pages_data)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [{"page_number": c["page_number"], "chunk_id": c["chunk_id"]} for c in chunks]
        ids = [c["chunk_id"] for c in chunks]

        # Compute embeddings
        embeddings = self.embedding_service.embed_documents(texts)

        if self.chroma_client:
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.doc_id,
                    metadata={"hnsw:space": "cosine"}
                )
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"ChromaDB indexing error: {e}")
                self.collection = None

        return len(chunks)

    def retrieve(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieves top_k most relevant chunks for the query."""
        if not self.collection:
            return []

        query_embedding = self.embedding_service.embed_query(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else []
                for i in range(len(docs)):
                    retrieved.append({
                        "text": docs[i],
                        "page_number": metas[i]["page_number"] if i < len(metas) else 1
                    })
            return retrieved
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Executes RAG pipeline to answer user question using context & returns answer with page citations.
        """
        if not question or not question.strip():
            return {"answer": "Please ask a question about the PDF.", "citations": []}

        relevant_chunks = self.retrieve(question, top_k=config.TOP_K_RESULTS)

        if not relevant_chunks:
            return {
                "answer": "I could not find relevant information in the uploaded PDF document to answer your question.",
                "citations": []
            }

        # Build context string with page tags
        context_parts = []
        cited_pages = set()
        for chunk in relevant_chunks:
            p_num = chunk["page_number"]
            cited_pages.add(p_num)
            context_parts.append(f"[Excerpt from PDF Page {p_num}]:\n{chunk['text']}")

        context_str = "\n\n".join(context_parts)
        pages_list_str = ", ".join(f"Page {p}" for p in sorted(cited_pages))

        if not self.api_key or not self.groq_client:
            # Fallback mock answer with citations if API key is not set
            fallback_ans = (
                f"Based on the PDF document ({pages_list_str}), here is what was found:\n\n"
                f"**Excerpt Summary**:\n{relevant_chunks[0]['text'][:300]}...\n\n"
                f"📍 *Sources / Page Citations*: {pages_list_str}\n\n"
                f"*(Note: Configure GROQ_API_KEY in sidebar or `.env` for complete LLM answer synthesis.)*"
            )
            self.chat_history.append({"user": question, "assistant": fallback_ans})
            return {"answer": fallback_ans, "citations": sorted(cited_pages)}

        prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, question=question)

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            answer = response.choices[0].message.content.strip()

            # Ensure page citations exist in the response
            if "Page" not in answer and cited_pages:
                answer += f"\n\n📍 **Page Citations**: {pages_list_str}"

            self.chat_history.append({"user": question, "assistant": answer})
            return {"answer": answer, "citations": sorted(cited_pages)}

        except Exception as e:
            err_msg = str(e)
            error_ans = f"⚠️ **Error generating answer**: {err_msg}\n\nRetrieved Context Sources: {pages_list_str}"
            return {"answer": error_ans, "citations": sorted(cited_pages)}
