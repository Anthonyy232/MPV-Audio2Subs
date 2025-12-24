"""Backward compatible entry point for the AI Subtitle Service.

This file is kept for compatibility with the Lua script.
The actual implementation is in src/audio2subs/
"""

import sys
import os

# Add src directory to path for package import
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from audio2subs.__main__ import main

if __name__ == "__main__":
    sys.exit(main())