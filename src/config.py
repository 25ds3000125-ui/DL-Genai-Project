GENRES = ["blues","classical","country","disco","hiphop",
          "jazz","metal","pop","reggae","rock"]

SR = 16000
DURATION = 10
SAMPLES = SR * DURATION

EPOCHS = 5
LR = 2e-5
BATCH_SIZE = 8

BASE_PATH = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"

TRAIN_PATH = f"{BASE_PATH}/genres_stems"
NOISE_PATH = f"{BASE_PATH}/ESC-50-master/audio"
TEST_PATH = f"{BASE_PATH}/mashups"
TEST_CSV = f"{BASE_PATH}/test.csv"