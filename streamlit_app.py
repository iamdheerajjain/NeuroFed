import os
import sys

# Add project root to Python path for Streamlit Cloud
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.web_app import main

if __name__ == "__main__":
    main()


