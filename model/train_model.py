import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

df = pd.read_csv('dataset_job_scraping.csv')
df.columns = ['No', 'Job_title', 'Job_Company', 'Location', 'Status', 'Time Period']

job_titles = df['Job_title'].tolist()
job_companies = df['Job_Company'].tolist()
labels = [0, 1, 2, 3, 4] * (len(job_titles) // 5)

small_job_titles = job_titles[1:50]
small_job_companies = job_companies[1:50]
small_labels = labels[1:50]
combined_texts = [f"{title} {company}" for title, company in zip(small_job_titles, small_job_companies)]

train_texts, val_texts, train_labels, val_labels = train_test_split(combined_texts, small_labels, test_size=0.2)

class JobDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
def train_and_save_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

    train_dataset = JobDataset(train_texts, train_labels, tokenizer)
    val_dataset = JobDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

def load_model(model_path="./trained_model"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model

if __name__ == "__main__":
    train_and_save_model()
