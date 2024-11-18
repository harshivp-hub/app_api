# Step 1: Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Step 2: Load and inspect the data
data_path = 'D:\\Semester - 7\\Capstone Project\\Capstone\\wellnessapp\\assets\\database\\data.csv'
df = pd.read_csv(data_path)

# Check data structure
print(df.head())

# Step 3: Preprocess the data
# Extract the answer text and create labels for emotions and topics
text_data = df['Answer'].astype(str)
emotion_columns = [col for col in df.columns if 'f1' in col and col.endswith('.raw')]
topic_columns = [col for col in df.columns if 't1' in col and col.endswith('.raw')]

# Combine emotion and topic labels for multi-label classification
df['labels'] = df[emotion_columns + topic_columns].values.tolist()

# Convert multi-label data into a multi-hot format
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['labels'])

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Step 4: Create a PyTorch Dataset class
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# Step 5: Initialize the BERT tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

# Step 6: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 7: Load the BERT model for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(mlb.classes_))

# Run the model on CPU
device = 'cpu'
model = model.to(device)

# Step 8: Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Step 9: Training loop
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validate after each epoch
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

        avg_val_loss = total_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
        model.train()

# Step 10: Run training
train_model(model, train_loader, val_loader, optimizer, scheduler)

# Save the model after training
model.save_pretrained('emotion_bert_model')
tokenizer.save_pretrained('emotion_bert_model')