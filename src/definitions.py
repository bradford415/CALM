"""
definitions.py

Used to store the directory paths. These paths are used
in the python scripts in the source directory

I was orginally running the program wrongly so I created this file.
It makes the code cleaner but probably isn't neccessary
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__)) # Source Directory
ROOT_DIR = os.path.join(SRC_DIR, "..")               # Root Directory
INPUT_DIR = os.path.join(ROOT_DIR,"input")           # Input Directory
OUTPUT_DIR = os.path.join(ROOT_DIR,"output")         # Output Directory
