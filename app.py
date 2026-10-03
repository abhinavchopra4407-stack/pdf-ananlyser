import os
import sys
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ui.components import build_app, CUSTOM_CSS
import gradio as gr


def main():
    """Main entry point to launch the PDF Query Assistant application."""
    print("==================================================")
    print("Starting PDF Query Assistant local web app...")
    print(f"Server running at: http://{config.HOST}:{config.PORT}")
    print("==================================================")
    sys.stdout.flush()

    demo = build_app()
    share_url = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    demo.launch(
        server_name=config.HOST,
        server_port=config.PORT,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        share=share_url,
        show_error=True
    )






if __name__ == "__main__":
    main()
