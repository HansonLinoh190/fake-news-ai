import os
import sys

# Resolve project path and imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "src"))
sys.path.append(os.path.join(base_dir, "app"))

from app.app import app
