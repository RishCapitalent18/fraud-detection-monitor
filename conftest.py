"""Ensures the repo root is on sys.path so `pytest` finds the fraud_monitor package."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
