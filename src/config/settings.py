# Configuration settings for ocean plastic detection project
"""
Application settings and configuration management.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

# Model settings
MODEL_INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data settings
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# Logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_LEVEL = "INFO"
