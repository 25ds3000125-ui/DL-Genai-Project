import torch.nn as nn
from transformers import ASTForAudioClassification

# Simple 3-layer Convolutional Neural Network for processing spectrogram images
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

# AST Approach
# Uses a pre-trained Transformer model from MIT to process audio as a sequence of patches
def get_ast_model(num_labels=10):
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    return model
