import gradio as gr
import torch
import librosa
import numpy as np
import random
import os
from transformers import AutoFeatureExtractor, ASTForAudioClassification

class CFG:
    AST_SR = 16000
    AST_DURATION = 10
    AST_SAMPLES = AST_SR * AST_DURATION

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Load the weights
if os.path.exists("ast_v2.pth"):
    model.load_state_dict(torch.load("ast_v2.pth", map_location=device))
else:
    print("WARNING: ast_v2.pth not found. The model will run with untrained weights.")

model.to(device)
model.eval()

# --- Functions ---
def load_audio(path, sr, samples):
    y, _ = librosa.load(path, sr=sr)
    if len(y) < samples:
        y = np.pad(y, (0, samples - len(y)))
    return y[:samples]

def extract(audio):
    return extractor(audio, sampling_rate=CFG.AST_SR, return_tensors="pt")["input_values"].squeeze(0)

# --- Prediction Wrapper ---
def predict_genre(audio_path):
    if audio_path is None:
        return "Please upload an audio file."
    
    try:
        y = load_audio(audio_path, CFG.AST_SR, CFG.AST_SAMPLES * 2) 
        y = y / (np.max(np.abs(y)) + 1e-9)

        preds = []
        n = 5 
        
        for _ in range(n):
            if len(y) <= CFG.AST_SAMPLES:
                start = 0
                crop = np.pad(y, (0, CFG.AST_SAMPLES - len(y)))
            else:
                start = random.randint(0, len(y) - CFG.AST_SAMPLES)
                crop = y[start:start + CFG.AST_SAMPLES]

            x = extract(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x).logits
                probs = torch.softmax(logits, dim=1)
                preds.append(probs.cpu().numpy()[0])

        mean_preds = np.mean(preds, axis=0)
        result_dict = {GENRES[i]: float(mean_preds[i]) for i in range(len(GENRES))}
        return result_dict
        
    except Exception as e:
        return {"Error": 0.0}

sample_audio_files = [
    ["song0001.wav"],
    ["song0002.wav"]
]

# --- Gradio UI ---
interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload Audio Song/Mashup"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Genre"),
    title="Messy Mashup Genre Classifier",
    description="Upload an audio file to predict its genre using the AST model, or click one of the examples below to test it out.",
    examples=sample_audio_files, 
    flagging_mode="never" 
)

if __name__ == "__main__":
    interface.launch()
