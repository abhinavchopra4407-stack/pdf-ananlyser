import gradio as gr
from typing import Dict, List, Any, Optional, Tuple
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from services.pdf_processor import PDFProcessor
from services.summarizer import SummarizerService
from services.rag_pipeline import RAGPipeline
from services.export_service import ExportService


CUSTOM_CSS = """
/* App styling for PDF Query Assistant */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.app-header {
    text-align: center;
    padding: 1.2rem;
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.app-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.3rem !important;
    color: #60A5FA !important;
}
.app-subtitle {
    font-size: 1.05rem !important;
    color: #94A3B8 !important;
    font-weight: 400 !important;
}
.meta-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
}
.warning-box {
    background-color: #FEF2F2;
    border-left: 4px solid #EF4444;
    padding: 10px;
    border-radius: 4px;
    color: #991B1B;
}
.summary-container {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 20px;
    min-height: 380px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.all-summaries-container {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 20px;
    max-height: 550px;
    overflow-y: auto;
}
"""


class DocumentSession:
    """Class to manage current document state cleanly without Gradio state serialization bugs."""
    def __init__(self):
        self.file_path: str = ""
        self.filename: str = ""
        self.pages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.current_page: int = 1
        self.total_pages: int = 1

    def reset(self):
        self.file_path = ""
        self.filename = ""
        self.pages = []
        self.metadata = {}
        self.current_page = 1
        self.total_pages = 1

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        self.reset()
        self.file_path = file_path
        processor = PDFProcessor(file_path)
        self.metadata = processor.process()
        self.pages = processor.pages_data
        self.filename = self.metadata["filename"]
        self.total_pages = self.metadata["total_pages"]
        self.current_page = 1
        return self.metadata


def build_app():
    # Instantiate backend document session & services
    session = DocumentSession()
    summarizer = SummarizerService()
    rag_pipeline = RAGPipeline()

    with gr.Blocks(title="PDF Query Assistant") as demo:
        
        # Header
        with gr.Row():
            gr.HTML("""
                <div class="app-header">
                    <h1 class="app-title">📚 PDF Query Assistant</h1>
                    <p class="app-subtitle">Page-wise Summarization & RAG System for Textbooks & Lengthy PDFs</p>
                </div>
            """)

        with gr.Row():
            # ================= LEFT SIDEBAR =================
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")
                pdf_uploader = gr.File(
                    label="Upload PDF Textbook",
                    file_types=[".pdf"],
                    file_count="single",
                    elem_id="pdf_uploader"
                )

                # API Key Settings
                with gr.Accordion("⚙️ Groq API & Model Settings", open=False):
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="gsk_...",
                        value=config.GROQ_API_KEY,
                        type="password"
                    )
                    model_selector = gr.Dropdown(
                        label="LLM Model",
                        choices=["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"],
                        value=config.GROQ_MODEL
                    )
                    save_config_btn = gr.Button("Save API Settings", size="sm")
                    config_status = gr.Markdown("")

                # Metadata Info Card
                meta_display = gr.Markdown(
                    "**Status**: No document uploaded yet.\n\n*Upload a PDF textbook to start page-wise summarization.*",
                    elem_classes=["meta-card"]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📖 Page Navigation")
                
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Prev", interactive=False, size="sm")
                    page_selector = gr.Dropdown(
                        label="Jump to Page",
                        choices=["Page 1"],
                        value="Page 1",
                        interactive=False
                    )
                    next_btn = gr.Button("Next ➡️", interactive=False, size="sm")

                with gr.Row():
                    regen_btn = gr.Button("🔄 Regenerate Page Summary", interactive=False, variant="secondary", size="sm")
                    
                search_input = gr.Textbox(
                    label="🔍 Search Summaries",
                    placeholder="Search keywords (e.g. supervised learning)...",
                    interactive=False
                )
                search_results_display = gr.Markdown("")

            # ================= MAIN WORKSPACE TABS =================
            with gr.Column(scale=3):
                with gr.Tabs():
                    
                    # ---------------- TAB 1: Page-wise Summaries ----------------
                    with gr.Tab("📖 Page Summaries"):
                        with gr.Row():
                            page_header = gr.Markdown("### PDF Page 1 — Summary")
                        
                        summary_output = gr.Markdown(
                            "Upload a PDF to view page-wise summaries.",
                            elem_classes=["summary-container"]
                        )
                        
                        with gr.Accordion("📜 View Original Page Text", open=False):
                            original_text_display = gr.Markdown("*Original extracted page text will appear here.*")

                        gr.Markdown("---")
                        with gr.Accordion("📋 View All Page Summaries Stacked (Full Document)", open=False):
                            all_summaries_display = gr.Markdown(
                                "*All page summaries will be listed here after processing.*",
                                elem_classes=["all-summaries-container"]
                            )

                    # ---------------- TAB 2: RAG AI Chat ----------------
                    with gr.Tab("💬 Chat with PDF (RAG)"):
                        gr.Markdown("### Ask Questions About the Uploaded Textbook")
                        chatbot = gr.Chatbot(
                            label="Document Assistant Chat",
                            height=450,
                            avatar_images=None
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Ask a question...",
                                placeholder="What is the main topic of Chapter 2?",
                                scale=4,
                                show_label=False
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                            clear_chat_btn = gr.Button("Clear Chat 🗑️", scale=1)

                        gr.Markdown("**Example Questions:**")
                        with gr.Row():
                            ex1 = gr.Button("Explain this document in simple words", size="sm")
                            ex2 = gr.Button("What are the main concepts in this PDF?", size="sm")
                            ex3 = gr.Button("Explain key terminology and definitions", size="sm")
                            ex4 = gr.Button("Give me important exam questions", size="sm")

                    # ---------------- TAB 3: Chapter & Full Doc Summaries ----------------
                    with gr.Tab("📚 Chapter & Revision Sheet"):
                        gr.Markdown("### Range & Complete Document Summarization")
                        
                        with gr.Row():
                            start_page_input = gr.Number(label="Start Page", value=1, precision=0, min_width=100)
                            end_page_input = gr.Number(label="End Page", value=5, precision=0, min_width=100)
                            gen_range_btn = gr.Button("Generate Range Summary", variant="secondary")

                        gr.Markdown("---")
                        gen_doc_btn = gr.Button("✨ Generate Full Document Summary & Revision Sheet", variant="primary")
                        
                        doc_summary_output = gr.Markdown("Click above to generate comprehensive textbook summary and master revision sheet.")

                    # ---------------- TAB 4: Export & Download ----------------
                    with gr.Tab("📥 Export & Download"):
                        gr.Markdown("### Download Generated Summaries")
                        gr.Markdown("Export page-wise summaries and revision guide in your preferred format.")
                        
                        with gr.Row():
                            export_txt_btn = gr.Button("📄 Download as TXT", variant="secondary")
                            export_md_btn = gr.Button("📝 Download as Markdown", variant="secondary")
                            export_pdf_btn = gr.Button("📕 Download as PDF", variant="primary")
                        
                        export_file_output = gr.File(label="Generated File Download", interactive=False)
                        export_status = gr.Markdown("")


        # ================= EVENT HANDLERS & CALLBACKS =================

        def on_save_config(api_key, model_name):
            summarizer.update_credentials(api_key, model_name)
            rag_pipeline.update_credentials(api_key, model_name)
            return "✅ API Settings saved successfully!"

        save_config_btn.click(
            fn=on_save_config,
            inputs=[api_key_input, model_selector],
            outputs=[config_status]
        )

        def render_all_summaries():
            blocks = []
            for p in session.pages:
                p_num = p["page_number"]
                s_text = summarizer.summary_cache.get(p_num, "*No summary generated.*")
                blocks.append(f"## 📄 PDF Page {p_num} — Summary\n\n{s_text}")
            return "\n\n---\n\n".join(blocks)

        def render_page(p_num: int, force_regen: bool = False):
            if not session.pages or p_num < 1 or p_num > session.total_pages:
                return "No data.", "*No text.*", "### PDF Page — Summary", gr.update(interactive=False), gr.update(interactive=False), "Page 1"

            session.current_page = p_num
            p_data = session.pages[p_num - 1]
            p_text = p_data["text"]
            summary = summarizer.generate_page_summary(p_num, p_text, force_regenerate=force_regen)

            header = f"### PDF Page {p_num} of {session.total_pages} — Summary"
            prev_active = (p_num > 1)
            next_active = (p_num < session.total_pages)
            page_str = f"Page {p_num}"

            return summary, p_text, header, gr.update(interactive=prev_active), gr.update(interactive=next_active), page_str

        def on_file_upload(file, progress=gr.Progress(track_tqdm=True)):
            if file is None:
                session.reset()
                return (
                    "**Status**: No document uploaded.",
                    "Upload a PDF to view page-wise summaries.",
                    "*No text extracted.*",
                    "*No summaries available.*",
                    gr.update(interactive=False),
                    gr.update(choices=["Page 1"], value="Page 1", interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "### PDF Page 1 — Summary",
                    []
                )

            try:
                progress(0.05, desc="Extracting text from PDF...")
                metadata = session.load_pdf(file.name)
                total_pages = session.total_pages

                progress(0.15, desc="Indexing vector embeddings for RAG...")
                chunks_count = rag_pipeline.build_vector_index(session.pages, session.filename)
                summarizer.clear_cache()

                # Pre-generate summaries for all pages with progress tracking
                for idx, p in enumerate(session.pages):
                    p_num = p["page_number"]
                    pct = 0.2 + (0.8 * ((idx + 1) / total_pages))
                    progress(pct, desc=f"Generating Summary for Page {p_num} of {total_pages}...")
                    summarizer.generate_page_summary(p_num, p["text"])

                p1_summary, p1_text, header_text, prev_act, next_act, p1_str = render_page(1)
                all_summaries_md = render_all_summaries()

                status_md = (
                    f"### 📋 Document Info\n"
                    f"- **Filename**: `{metadata['filename']}`\n"
                    f"- **Total Pages**: {total_pages}\n"
                    f"- **File Size**: {metadata['file_size_formatted']}\n"
                    f"- **RAG Chunks Indexed**: {chunks_count}\n"
                    f"- **Page Summaries Ready**: {len(summarizer.summary_cache)} / {total_pages}\n"
                )
                if metadata["is_scanned"]:
                    status_md += "\n<div class='warning-box'>⚠️ <b>Notice</b>: This PDF appears to be scanned or image-based. OCR may be required.</div>"
                else:
                    status_md += "\n✅ All page summaries generated successfully!"

                page_choices = [f"Page {i}" for i in range(1, total_pages + 1)]

                return (
                    status_md,
                    p1_summary,
                    p1_text,
                    all_summaries_md,
                    prev_act,
                    gr.update(choices=page_choices, value="Page 1", interactive=True),
                    next_act,
                    gr.update(interactive=True), # regen_btn
                    gr.update(interactive=True), # search_input
                    header_text,
                    []
                )
            except Exception as e:
                session.reset()
                err_status = f"❌ **Error processing PDF**: {str(e)}"
                return (
                    err_status,
                    f"Error: {str(e)}",
                    "*Extraction failed.*",
                    "*Processing failed.*",
                    gr.update(interactive=False),
                    gr.update(choices=["Page 1"], value="Page 1", interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "### Error",
                    []
                )

        pdf_uploader.change(
            fn=on_file_upload,
            inputs=[pdf_uploader],
            outputs=[
                meta_display, summary_output, original_text_display, all_summaries_display,
                prev_btn, page_selector, next_btn, regen_btn, search_input,
                page_header, chatbot
            ]
        )

        def on_prev_click():
            new_p = max(1, session.current_page - 1)
            summary, p_text, header, prev_act, next_act, p_str = render_page(new_p)
            return summary, p_text, header, prev_act, next_act, p_str

        def on_next_click():
            new_p = min(session.total_pages, session.current_page + 1)
            summary, p_text, header, prev_act, next_act, p_str = render_page(new_p)
            return summary, p_text, header, prev_act, next_act, p_str

        prev_btn.click(
            fn=on_prev_click,
            inputs=[],
            outputs=[summary_output, original_text_display, page_header, prev_btn, next_btn, page_selector]
        )

        next_btn.click(
            fn=on_next_click,
            inputs=[],
            outputs=[summary_output, original_text_display, page_header, prev_btn, next_btn, page_selector]
        )

        def on_page_select(selected_str):
            if not selected_str or not session.pages:
                return "No data.", "*No text.*", "### Page 1", gr.update(interactive=False), gr.update(interactive=False), "Page 1"
            p_num = int(selected_str.replace("Page ", "").strip())
            summary, p_text, header, prev_act, next_act, p_str = render_page(p_num)
            return summary, p_text, header, prev_act, next_act, p_str

        page_selector.input(
            fn=on_page_select,
            inputs=[page_selector],
            outputs=[summary_output, original_text_display, page_header, prev_btn, next_btn, page_selector]
        )

        def on_regenerate():
            summary, p_text, header, prev_act, next_act, p_str = render_page(session.current_page, force_regen=True)
            all_summaries_md = render_all_summaries()
            return summary, all_summaries_md

        regen_btn.click(
            fn=on_regenerate,
            inputs=[],
            outputs=[summary_output, all_summaries_display]
        )

        def on_search_summaries(keyword):
            if not keyword or not keyword.strip() or not session.pages:
                return ""
            kw = keyword.lower().strip()
            matches = []
            for p_num, summary in summarizer.summary_cache.items():
                if kw in summary.lower():
                    matches.append(f"- **Page {p_num}**: Match found in summary text.")
            if not matches:
                return f"No matches found for keyword: *'{keyword}'*"
            return f"**Found matches on {len(matches)} page(s):**\n" + "\n".join(matches)

        search_input.change(
            fn=on_search_summaries,
            inputs=[search_input],
            outputs=[search_results_display]
        )

        # ---------------- RAG CHAT CALLBACKS ----------------

        def chat_respond(user_message, history):
            if not user_message or not user_message.strip():
                return "", history

            res = rag_pipeline.answer_question(user_message)
            answer = res["answer"]
            history.append((user_message, answer))
            return "", history

        send_btn.click(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        msg_input.submit(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        clear_chat_btn.click(
            fn=lambda: (rag_pipeline.chat_history.clear(), []),
            outputs=[msg_input, chatbot]
        )

        ex1.click(fn=lambda: "Explain this document in simple words", outputs=[msg_input])
        ex2.click(fn=lambda: "What are the main concepts in this PDF?", outputs=[msg_input])
        ex3.click(fn=lambda: "Explain key terminology and definitions", outputs=[msg_input])
        ex4.click(fn=lambda: "Give me important exam questions", outputs=[msg_input])

        # ---------------- CHAPTER / RANGE / DOC SUMMARY CALLBACKS ----------------

        def on_gen_range(start_p, end_p):
            if not session.pages:
                return "Please upload a PDF document first."
            start_p = int(start_p)
            end_p = int(end_p)
            return summarizer.generate_range_summary(start_p, end_p, session.pages)

        gen_range_btn.click(
            fn=on_gen_range,
            inputs=[start_page_input, end_page_input],
            outputs=[doc_summary_output]
        )

        def on_gen_doc_summary():
            if not session.pages:
                return "Please upload a PDF document first."
            filename = session.filename or "document.pdf"
            return summarizer.generate_document_summary(filename, session.pages)

        gen_doc_btn.click(
            fn=on_gen_doc_summary,
            inputs=[],
            outputs=[doc_summary_output]
        )

        # ---------------- EXPORT CALLBACKS ----------------

        def on_export(fmt):
            if not summarizer.summary_cache:
                return None, "⚠️ No page summaries available to export. Please process a document first."

            filename = session.filename or "document.pdf"
            doc_sum = summarizer.doc_summary_cache or ""

            if fmt == "txt":
                file_path = ExportService.export_to_txt(filename, summarizer.summary_cache, doc_sum)
            elif fmt == "md":
                file_path = ExportService.export_to_markdown(filename, summarizer.summary_cache, doc_sum)
            elif fmt == "pdf":
                file_path = ExportService.export_to_pdf(filename, summarizer.summary_cache, doc_sum)
            else:
                return None, "Invalid format."

            return file_path, f"✅ Export ready: `{Path(file_path).name}`"

        export_txt_btn.click(
            fn=lambda: on_export("txt"),
            inputs=[],
            outputs=[export_file_output, export_status]
        )
        export_md_btn.click(
            fn=lambda: on_export("md"),
            inputs=[],
            outputs=[export_file_output, export_status]
        )
        export_pdf_btn.click(
            fn=lambda: on_export("pdf"),
            inputs=[],
            outputs=[export_file_output, export_status]
        )

    return demo
