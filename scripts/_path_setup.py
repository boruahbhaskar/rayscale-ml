# scripts/_path_setup.py
import sys
from pathlib import Path

def setup_paths():
    """Setup Python paths for the project."""
    # Add project root to sys.path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

# Call it immediately when imported
setup_paths()