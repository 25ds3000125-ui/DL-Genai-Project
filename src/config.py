#!/usr/bin/env python
# coding: utf-8

import torch

# Configuration & Hyperparameters
# Defines sampling rates, durations, and training settings for CNN and AST models
class CFG:
    SR = 22050
    AST_SR = 16000
    DURATION = 5
    AST_DURATION = 10
    SAMPLES = SR * DURATION
    AST_SAMPLES = AST_SR * AST_DURATION
    N_MELS = 128
    MEL_WIDTH = 216
    BATCH_SIZE = 16
    AST_BATCH = 8
    EPOCHS = 4
    AST_EPOCHS = 8
    LR = 1e-3
    AST_LR = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"
TRAIN_PATH = f"{BASE_PATH}/genres_stems"
NOISE_PATH = f"{BASE_PATH}/ESC-50-master/audio"
TEST_PATH = f"{BASE_PATH}/mashups"
TEST_CSV = f"{BASE_PATH}/test.csv"

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
label2idx = {g:i for i,g in enumerate(GENRES)}
idx2label = {i:g for g,i in label2idx.items()}
