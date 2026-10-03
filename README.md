# 📚 PDF Query Assistant — Page-wise Summarization & RAG System

An intelligent, full-stack Python application built with **Gradio**, **LangChain**, **Groq API**, **ChromaDB**, and **HuggingFace sentence-transformers**. Designed specifically for understanding lengthy PDF textbooks by generating simple, structured, page-by-page summaries and answering questions with exact page citations.

---

## 🌟 Key Features

1. **Page-wise PDF Summarization**:
   - Generates structured, beginner-friendly explanations for *every page*.
   - Includes **Topic**, **Simple Explanation**, **Key Points**, **In Simple Words** analogy, and **Important Terms**.
   - Preserves 1-indexed page numbers.
   - Page caching ensures instant navigation between previously generated pages without unnecessary LLM calls.
   - Page selector, Previous/Next navigation, keyword search, and single-page summary regeneration.

2. **AI-Powered RAG Chatbot with Page Citations**:
   - Vector indexing using ChromaDB and HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
   - Answers questions strictly based on uploaded PDF contents using Groq's `llama-3.1-8b-instant`.
   - Every answer cites exact source page numbers.
   - Falls back gracefully when information is not present in the document.

3. **Chapter, Range & Complete Document Summaries**:
   - Range summarizer for selected page spans (e.g. Pages 10–15).
   - Generates document-wide executive summaries and **Master Revision Sheets**.

4. **Multi-Format Export**:
   - Download generated summaries in **TXT**, **Markdown**, or **PDF** format.

---

## 🚀 Setup & Installation (Windows)

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Create and Activate Virtual Environment
Open PowerShell or Command Prompt inside the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Configuration (.env)

1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` and configure your **Groq API Key**:
   - Obtain a free Groq API Key from [https://console.groq.com/keys](https://console.groq.com/keys).
   - Set your key in `.env`:
     ```env
     GROQ_API_KEY=gsk_your_actual_groq_api_key_here
     GROQ_MODEL=llama-3.1-8b-instant
     ```
   *(Note: You can also update the API Key live inside the application's sidebar UI settings).*

---

## 🏃 Running the Application

Start the local web application:

```powershell
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:7860`**

---

## 🧪 Running Automated Tests

Run the pytest suite to verify text extraction, page preservation, RAG chunking, LLM mocking, and file exports:

```powershell
pytest -v
```

---

## 🛠️ Project Architecture

```
pdf-query-assistant/
├── app.py                   # Main Gradio application runner
├── config.py                # Central configuration & env loader
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable template
├── README.md                # Documentation & instructions
├── .gitignore               # Git ignore rules
├── services/
│   ├── pdf_processor.py     # Text extraction & page metadata
│   ├── summarizer.py        # LLM summarization pipeline & caching
│   ├── rag_pipeline.py      # ChromaDB vector store & Q&A RAG
│   ├── embeddings.py        # SentenceTransformers embeddings
│   └── export_service.py    # TXT, MD, and PDF export generator
├── ui/
│   └── components.py        # Gradio modern web UI layout & state
├── utils/
│   └── helpers.py           # Helper validation & formatting
└── tests/
    ├── test_pdf_processor.py
    ├── test_summarizer.py
    └── test_rag_pipeline.py
```

---

## ❓ Troubleshooting

- **API Key Error**: Ensure `GROQ_API_KEY` is set in `.env` or in the application sidebar accordion under *Groq API Settings*.
- **Scanned PDF Warning**: If a PDF contains image scans rather than selectable text, the app will display a notification that OCR is required.
- **Port Conflict**: If port `7860` is in use, edit `PORT=7861` in `.env`.
