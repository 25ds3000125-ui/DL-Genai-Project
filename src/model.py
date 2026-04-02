from transformers import ASTForAudioClassification

def get_model():
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    return model