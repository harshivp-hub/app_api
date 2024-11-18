import sys
import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Redirect TensorFlow logging to stderr
# tf.get_logger().setLevel('ERROR')

# Force CPU usage
device = torch.device("cpu")

# Load the trained model and tokenizer
model_path = 'roberta_emotion_model'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)  # Move model to CPU
model.eval()

# Define labels and thresholds
emotion_labels = [
    'afraid', 'angry', 'anxious', 'ashamed', 'awkward', 'bored', 'calm',
    'confused', 'disgusted', 'excited', 'frustrated', 'happy', 'jealous',
    'nostalgic', 'proud', 'sad', 'satisfied', 'surprised'
]
topic_labels = [
    'exercise', 'family', 'food', 'friends', 'god', 'health', 'love',
    'recreation', 'school', 'sleep', 'work'
]
optimal_thresholds = [
    0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.2, 0.2, 0.35, 0.2, 0.2, 0.45, 0.5,
    0.15, 0.25, 0.3, 0.35, 0.5, 0.2, 0.25, 0.25, 0.2, 0.25, 0.1, 0.25, 0.1,
    0.15, 0.25, 0.25
]

# Process input text
input_text = sys.argv[1]
inputs = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = inputs['input_ids'].to(device)  # Move input tensors to CPU
attention_mask = inputs['attention_mask'].to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten()  # Move results to CPU

# Apply thresholds
emotion_preds = (probs[:len(emotion_labels)] >= np.array(optimal_thresholds[:len(emotion_labels)])).astype(int)
topic_preds = (probs[len(emotion_labels):] >= np.array(optimal_thresholds[len(emotion_labels):])).astype(int)

emotions = [label for label, pred in zip(emotion_labels, emotion_preds) if pred]
topics = [label for label, pred in zip(topic_labels, topic_preds) if pred]

# Output results as JSON
result = json.dumps({'emotions': emotions, 'topics': topics})
#print(result)
print("Input text:", input_text)  # Log input text
print("Emotion results:", result)  # Log emotion detection results