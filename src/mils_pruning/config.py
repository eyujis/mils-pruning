import os
import sys
from pathlib import Path

# Get the absolute path to the root of the repo (assumes 'src/' is in path)
this_file = Path(__file__).resolve()

# Go up until we find the project root (the parent of 'src')
PROJECT_ROOT = this_file
while PROJECT_ROOT.name != "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
PROJECT_ROOT = PROJECT_ROOT.parent

# Paths used throughout the project
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "saved_weights"
