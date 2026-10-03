import streamlit as st
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from services.pdf_processor import PDFProcessor
from services.summarizer import SummarizerService
from services.rag_pipeline import RAGPipeline
from services.export_service import ExportService

# Page configuration
st.set_page_config(
    page_title="PDF Query Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Safely initialize Streamlit session state dictionary keys."""
    if "summarizer" not in st.session_state:
        st.session_state["summarizer"] = SummarizerService()

    if "rag_pipeline" not in st.session_state:
        st.session_state["rag_pipeline"] = RAGPipeline()

    if "pdf_data" not in st.session_state:
        st.session_state["pdf_data"] = None

    if "pages_data" not in st.session_state:
        st.session_state["pages_data"] = []

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if "quiz_data" not in st.session_state:
        st.session_state["quiz_data"] = ""


def main():
    init_session_state()

    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1.4rem;
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .main-title {
            color: #60A5FA;
            font-size: 2.3rem;
            font-weight: 700;
        }
        .badge {
            background-color: #DBEAFE;
            color: #1E40AF;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        </style>
        <div class="main-header">
            <div class="main-title">📚 PDF Query Assistant</div>
            <p style="color: #94A3B8; margin-top: 5px;">Advanced Page-wise Summarization, RAG System & Exam Practice for Textbooks</p>
        </div>
    """, unsafe_allow_html=True)

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF Textbook", type=["pdf"])

        with st.expander("⚙️ LLM & RAG Advanced Settings", expanded=False):
            api_key = st.text_input("Groq API Key", value=config.GROQ_API_KEY, type="password")
            model_name = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
            top_k = st.slider("RAG Context Chunks (Top-K)", min_value=1, max_value=10, value=config.TOP_K_RESULTS)
            
            if st.button("Save Settings"):
                st.session_state["summarizer"].update_credentials(api_key, model_name)
                st.session_state["rag_pipeline"].update_credentials(api_key, model_name)
                st.success("Advanced settings saved!")

        if uploaded_file is not None:
            curr_pdf = st.session_state.get("pdf_data")
            if curr_pdf is None or curr_pdf.get("filename") != uploaded_file.name:
                with st.spinner("Processing PDF textbook & indexing vector database..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        processor = PDFProcessor(tmp_path)
                        metadata = processor.process()
                        metadata["filename"] = uploaded_file.name
                        
                        st.session_state["pdf_data"] = metadata
                        st.session_state["pages_data"] = processor.pages_data
                        st.session_state["current_page"] = 1
                        st.session_state["quiz_data"] = ""

                        # Build RAG vector index
                        st.session_state["rag_pipeline"].build_vector_index(processor.pages_data, uploaded_file.name)
                        st.session_state["summarizer"].clear_cache()

                        # Pre-generate summaries for all pages
                        progress_bar = st.progress(0, text="Generating page summaries...")
                        total_p = metadata["total_pages"]
                        for i, p in enumerate(processor.pages_data):
                            progress_bar.progress((i + 1) / total_p, text=f"Summarizing Page {p['page_number']} of {total_p}...")
                            st.session_state["summarizer"].generate_page_summary(p["page_number"], p["text"])
                        progress_bar.empty()

                        st.sidebar.success("PDF processed & vector index ready!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

        # Metadata Card
        pdf_meta = st.session_state.get("pdf_data")
        if pdf_meta:
            st.info(f"**Filename**: {pdf_meta['filename']}\n\n"
                    f"**Total Pages**: {pdf_meta['total_pages']}\n\n"
                    f"**File Size**: {pdf_meta['file_size_formatted']}")
            if pdf_meta.get("is_scanned"):
                st.warning("⚠️ Scanned PDF detected. OCR may be required.")
            
            # Chapter Bookmarks Navigator if TOC present
            chapters = pdf_meta.get("chapters", [])
            if chapters:
                st.markdown("### 🔖 Table of Contents")
                ch_titles = [f"P.{c['page_number']}: {c['title']}" for c in chapters]
                selected_ch = st.selectbox("Jump to Chapter", ch_titles)
                if selected_ch:
                    p_target = int(selected_ch.split(":")[0].replace("P.", "").strip())
                    if p_target != st.session_state.get("current_page"):
                        st.session_state["current_page"] = p_target
                        st.rerun()
        else:
            st.info("Upload a PDF document to begin.")

    # ================= MAIN TABS =================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Page Summaries",
        "💬 Chat with PDF (RAG)",
        "🎯 Exam Quiz & Practice",
        "📚 Chapter & Revision Sheet",
        "📥 Export & Download"
    ])

    pages = st.session_state.get("pages_data", [])

    # ---------------- TAB 1: PAGE SUMMARIES ----------------
    with tab1:
        if not pages:
            st.info("Please upload a PDF textbook in the sidebar to view page summaries.")
        else:
            pdf_info = st.session_state.get("pdf_data", {})
            total_pages = pdf_info.get("total_pages", len(pages))
            curr_p = st.session_state.get("current_page", 1)
            
            # Navigation Bar
            col_prev, col_select, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("⬅️ Previous Page", disabled=(curr_p <= 1)):
                    st.session_state["current_page"] = max(1, curr_p - 1)
                    st.rerun()

            with col_select:
                selected_page_str = st.selectbox(
                    "Select Page",
                    options=[f"Page {i}" for i in range(1, total_pages + 1)],
                    index=max(0, min(curr_p - 1, total_pages - 1))
                )
                selected_num = int(selected_page_str.replace("Page ", "").strip())
                if selected_num != curr_p:
                    st.session_state["current_page"] = selected_num
                    st.rerun()

            with col_next:
                if st.button("Next Page ➡️", disabled=(curr_p >= total_pages)):
                    st.session_state["current_page"] = min(total_pages, curr_p + 1)
                    st.rerun()

            curr_p = st.session_state.get("current_page", 1)
            p_data = pages[curr_p - 1]
            
            st.subheader(f"PDF Page {curr_p} of {total_pages} — Summary")
            
            summary = st.session_state["summarizer"].generate_page_summary(curr_p, p_data["text"])
            st.markdown(summary)

            if st.button("🔄 Regenerate Page Summary"):
                summary = st.session_state["summarizer"].generate_page_summary(curr_p, p_data["text"], force_regenerate=True)
                st.rerun()

            with st.expander("📜 View Original Extracted Page Text", expanded=False):
                st.text_area("Extracted Text", value=p_data["text"], height=200, disabled=True)

            st.markdown("---")
            with st.expander("📋 View All Page Summaries Stacked (Full Document)", expanded=False):
                for p in pages:
                    p_n = p["page_number"]
                    s_t = st.session_state["summarizer"].summary_cache.get(p_n, "")
                    st.markdown(f"### PDF Page {p_n} — Summary\n\n{s_t}")
                    st.markdown("---")

    # ---------------- TAB 2: RAG CHAT ----------------
    with tab2:
        st.subheader("💬 Ask Questions About the PDF (RAG Pipeline)")
        
        # Example questions
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        q_clicked = None
        if col_q1.button("Explain in simple words"):
            q_clicked = "Explain this document in simple words"
        if col_q2.button("Main concepts"):
            q_clicked = "What are the main concepts in this PDF?"
        if col_q3.button("Key terminology"):
            q_clicked = "Explain key terminology and definitions"
        if col_q4.button("Exam questions"):
            q_clicked = "Give me important exam questions from this document"

        chat_msgs = st.session_state.get("chat_messages", [])
        for msg in chat_msgs:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Ask a question about the PDF...") or q_clicked

        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching vector index & synthesizing response..."):
                    res = st.session_state["rag_pipeline"].answer_question(prompt)
                    ans = res["answer"]
                    st.write(ans)

                    # Show cited pages badge
                    if res.get("citations"):
                        c_str = ", ".join(f"Page {p}" for p in res["citations"])
                        st.caption(f"📍 **Cited PDF Sources**: {c_str}")

                    st.session_state["chat_messages"].append({"role": "assistant", "content": ans})

    # ---------------- TAB 3: EXAM QUIZ & PRACTICE ----------------
    with tab3:
        st.subheader("🎯 Interactive Exam Quiz Generator")
        st.write("Generate practice multiple choice questions (MCQs) directly from the PDF page summaries to test your understanding.")

        if st.button("🚀 Generate 5 Practice MCQs from Textbook"):
            if pages:
                with st.spinner("Generating exam practice questions..."):
                    pdf_info = st.session_state.get("pdf_data", {})
                    quiz_md = st.session_state["summarizer"].generate_quiz(
                        pdf_info.get("filename", "document.pdf"),
                        pages
                    )
                    st.session_state["quiz_data"] = quiz_md
            else:
                st.warning("Please upload a PDF first.")

        if st.session_state.get("quiz_data"):
            st.markdown(st.session_state["quiz_data"])

    # ---------------- TAB 4: CHAPTER & DOC SUMMARY ----------------
    with tab4:
        st.subheader("📚 Section & Master Revision Sheet")
        
        col_s1, col_s2 = st.columns(2)
        start_p = col_s1.number_input("Start Page", min_value=1, value=1)
        end_p = col_s2.number_input("End Page", min_value=1, value=min(5, len(pages) or 5))

        if st.button("Generate Section Summary"):
            if pages:
                with st.spinner("Generating range summary..."):
                    res = st.session_state["summarizer"].generate_range_summary(int(start_p), int(end_p), pages)
                    st.markdown(res)
            else:
                st.warning("Please upload a PDF first.")

        st.markdown("---")
        if st.button("✨ Generate Complete Document Summary & Master Revision Sheet"):
            if pages:
                with st.spinner("Generating document-level revision guide..."):
                    pdf_info = st.session_state.get("pdf_data", {})
                    doc_res = st.session_state["summarizer"].generate_document_summary(
                        pdf_info.get("filename", "document.pdf"),
                        pages
                    )
                    st.markdown(doc_res)
            else:
                st.warning("Please upload a PDF first.")

    # ---------------- TAB 5: EXPORT & DOWNLOAD ----------------
    with tab5:
        st.subheader("📥 Export & Download Summaries")
        summarizer_inst = st.session_state.get("summarizer")
        if not pages or not summarizer_inst or not summarizer_inst.summary_cache:
            st.info("Upload a PDF and generate summaries to enable exports.")
        else:
            pdf_info = st.session_state.get("pdf_data", {})
            filename = pdf_info.get("filename", "document.pdf")
            doc_sum = summarizer_inst.doc_summary_cache or ""

            col_t, col_m, col_p = st.columns(3)

            with col_t:
                txt_path = ExportService.export_to_txt(filename, summarizer_inst.summary_cache, doc_sum)
                with open(txt_path, "r", encoding="utf-8") as f:
                    st.download_button("📄 Download TXT", data=f.read(), file_name=Path(txt_path).name, mime="text/plain")

            with col_m:
                md_path = ExportService.export_to_markdown(filename, summarizer_inst.summary_cache, doc_sum)
                with open(md_path, "r", encoding="utf-8") as f:
                    st.download_button("📝 Download Markdown", data=f.read(), file_name=Path(md_path).name, mime="text/markdown")

            with col_p:
                pdf_path = ExportService.export_to_pdf(filename, summarizer_inst.summary_cache, doc_sum)
                with open(pdf_path, "rb") as f:
                    st.download_button("📕 Download PDF", data=f.read(), file_name=Path(pdf_path).name, mime="application/pdf")


if __name__ == "__main__":
    main()
