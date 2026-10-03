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

# Initialize Session State
if "summarizer" not in st.session_state:
    st.session_state.summarizer = SummarizerService()

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None  # metadata dict

if "pages_data" not in st.session_state:
    st.session_state.pages_data = []

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


def main():
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .main-title {
            color: #60A5FA;
            font-size: 2.2rem;
            font-weight: 700;
        }
        </style>
        <div class="main-header">
            <div class="main-title">📚 PDF Query Assistant</div>
            <p style="color: #94A3B8; margin-top: 5px;">Page-wise Summarization & RAG System for Textbooks & Lengthy PDFs</p>
        </div>
    """, unsafe_allow_html=True)

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Upload PDF Textbook", type=["pdf"])

        with st.expander("⚙️ Groq API & Settings", expanded=False):
            api_key = st.text_input("Groq API Key", value=config.GROQ_API_KEY, type="password")
            model_name = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
            if st.button("Save API Settings"):
                st.session_state.summarizer.update_credentials(api_key, model_name)
                st.session_state.rag_pipeline.update_credentials(api_key, model_name)
                st.success("API Settings saved!")

        if uploaded_file is not None:
            # Save uploaded bytes to a temporary file
            if st.session_state.pdf_data is None or st.session_state.pdf_data.get("filename") != uploaded_file.name:
                with st.spinner("Processing PDF textbook & indexing vectors..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        processor = PDFProcessor(tmp_path)
                        metadata = processor.process()
                        metadata["filename"] = uploaded_file.name
                        
                        st.session_state.pdf_data = metadata
                        st.session_state.pages_data = processor.pages_data
                        st.session_state.current_page = 1

                        # Build RAG vector index
                        st.session_state.rag_pipeline.build_vector_index(processor.pages_data, uploaded_file.name)
                        st.session_state.summarizer.clear_cache()

                        # Pre-generate summaries for all pages
                        progress_bar = st.progress(0, text="Generating page summaries...")
                        total_p = metadata["total_pages"]
                        for i, p in enumerate(processor.pages_data):
                            progress_bar.progress((i + 1) / total_p, text=f"Summarizing Page {p['page_number']} of {total_p}...")
                            st.session_state.summarizer.generate_page_summary(p["page_number"], p["text"])
                        progress_bar.empty()

                        st.sidebar.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

        # Metadata Card
        if st.session_state.pdf_data:
            meta = st.session_state.pdf_data
            st.info(f"**Filename**: {meta['filename']}\n\n"
                    f"**Total Pages**: {meta['total_pages']}\n\n"
                    f"**File Size**: {meta['file_size_formatted']}")
            if meta.get("is_scanned"):
                st.warning("⚠️ Scanned PDF detected. OCR may be required.")
        else:
            st.info("Upload a PDF document to begin.")

    # ================= MAIN TABS =================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Page Summaries",
        "💬 Chat with PDF (RAG)",
        "📚 Chapter & Revision Sheet",
        "📥 Export & Download"
    ])

    # ---------------- TAB 1: PAGE SUMMARIES ----------------
    with tab1:
        if not st.session_state.pages_data:
            st.info("Please upload a PDF textbook in the sidebar to view page summaries.")
        else:
            total_pages = st.session_state.pdf_data["total_pages"]
            
            # Navigation Bar
            col_prev, col_select, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("⬅️ Previous Page", disabled=(st.session_state.current_page <= 1)):
                    st.session_state.current_page -= 1
                    st.rerun()

            with col_select:
                selected_page_str = st.selectbox(
                    "Jump to Page",
                    options=[f"Page {i}" for i in range(1, total_pages + 1)],
                    index=st.session_state.current_page - 1
                )
                selected_num = int(selected_page_str.replace("Page ", "").strip())
                if selected_num != st.session_state.current_page:
                    st.session_state.current_page = selected_num
                    st.rerun()

            with col_next:
                if st.button("Next Page ➡️", disabled=(st.session_state.current_page >= total_pages)):
                    st.session_state.current_page += 1
                    st.rerun()

            curr_p = st.session_state.current_page
            p_data = st.session_state.pages_data[curr_p - 1]
            
            st.subheader(f"PDF Page {curr_p} of {total_pages} — Summary")
            
            # Generate / retrieve summary
            summary = st.session_state.summarizer.generate_page_summary(curr_p, p_data["text"])
            st.markdown(summary)

            if st.button("🔄 Regenerate Page Summary"):
                summary = st.session_state.summarizer.generate_page_summary(curr_p, p_data["text"], force_regenerate=True)
                st.rerun()

            with st.expander("📜 View Original Extracted Page Text", expanded=False):
                st.text_area("Extracted Text", value=p_data["text"], height=200, disabled=True)

            st.markdown("---")
            with st.expander("📋 View All Page Summaries Stacked (Full Document)", expanded=False):
                for p in st.session_state.pages_data:
                    p_n = p["page_number"]
                    s_t = st.session_state.summarizer.summary_cache.get(p_n, "")
                    st.markdown(f"### PDF Page {p_n} — Summary\n\n{s_t}")
                    st.markdown("---")

    # ---------------- TAB 2: RAG CHAT ----------------
    with tab2:
        st.subheader("💬 Ask Questions About the PDF")
        
        # Example question buttons
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

        # Display Chat History
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Ask a question about the PDF...") or q_clicked

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking & searching document context..."):
                    res = st.session_state.rag_pipeline.answer_question(prompt)
                    ans = res["answer"]
                    st.write(ans)
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    # ---------------- TAB 3: CHAPTER & DOC SUMMARY ----------------
    with tab3:
        st.subheader("📚 Section & Revision Sheet Summaries")
        
        col_s1, col_s2 = st.columns(2)
        start_p = col_s1.number_input("Start Page", min_value=1, value=1)
        end_p = col_s2.number_input("End Page", min_value=1, value=min(5, len(st.session_state.pages_data) or 5))

        if st.button("Generate Section Summary"):
            if st.session_state.pages_data:
                with st.spinner("Generating range summary..."):
                    res = st.session_state.summarizer.generate_range_summary(int(start_p), int(end_p), st.session_state.pages_data)
                    st.markdown(res)
            else:
                st.warning("Please upload a PDF first.")

        st.markdown("---")
        if st.button("✨ Generate Complete Document Summary & Master Revision Sheet"):
            if st.session_state.pages_data:
                with st.spinner("Generating document-level revision guide..."):
                    doc_res = st.session_state.summarizer.generate_document_summary(
                        st.session_state.pdf_data["filename"],
                        st.session_state.pages_data
                    )
                    st.markdown(doc_res)
            else:
                st.warning("Please upload a PDF first.")

    # ---------------- TAB 4: EXPORT & DOWNLOAD ----------------
    with tab4:
        st.subheader("📥 Export & Download Summaries")
        if not st.session_state.pages_data or not st.session_state.summarizer.summary_cache:
            st.info("Upload a PDF and generate summaries to enable exports.")
        else:
            filename = st.session_state.pdf_data["filename"]
            doc_sum = st.session_state.summarizer.doc_summary_cache or ""

            col_t, col_m, col_p = st.columns(3)

            with col_t:
                txt_path = ExportService.export_to_txt(filename, st.session_state.summarizer.summary_cache, doc_sum)
                with open(txt_path, "r", encoding="utf-8") as f:
                    st.download_button("📄 Download TXT", data=f.read(), file_name=Path(txt_path).name, mime="text/plain")

            with col_m:
                md_path = ExportService.export_to_markdown(filename, st.session_state.summarizer.summary_cache, doc_sum)
                with open(md_path, "r", encoding="utf-8") as f:
                    st.download_button("📝 Download Markdown", data=f.read(), file_name=Path(md_path).name, mime="text/markdown")

            with col_p:
                pdf_path = ExportService.export_to_pdf(filename, st.session_state.summarizer.summary_cache, doc_sum)
                with open(pdf_path, "rb") as f:
                    st.download_button("📕 Download PDF", data=f.read(), file_name=Path(pdf_path).name, mime="application/pdf")


if __name__ == "__main__":
    main()
