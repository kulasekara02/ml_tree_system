"""
Launch the ML Tree System GUI Interface.
Run: python run_gui.py
"""
import os
import sys

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from interface.gui_app import main

if __name__ == "__main__":
    print("Starting ML Tree System GUI...")
    print("Loading models and initializing interface...")
    main()
