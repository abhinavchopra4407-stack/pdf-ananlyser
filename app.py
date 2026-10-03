import sys
from pathlib import Path

# Add project root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from streamlit_app import main

if __name__ == "__main__":
    main()
