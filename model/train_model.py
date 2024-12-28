import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Đọc file CSV
df = pd.read_csv('dataset_job_scraping.csv')

# Kiểm tra các cột có trong dataset
df.columns = ['No', 'Job_title', 'Job_Company', 'Location', 'Status', 'Time Period']  # Đặt tên các cột

# Lấy các cột cần thiết
job_titles = df['Job_title'].tolist()
job_companies = df['Job_Company'].tolist()
labels = [0, 1, 2, 3, 4] * (len(job_titles) // 5)  # Tạo labels giả định cho dữ liệu

# Đảm bảo rằng job_titles và job_companies có cùng số lượng phần tử
small_job_titles = job_titles[1:100]
small_job_companies = job_companies[1:100]
small_labels = labels[1:100]
combined_texts = [f"{title} {company}" for title, company in zip(small_job_titles, small_job_companies)]

train_texts, val_texts, train_labels, val_labels = train_test_split(combined_texts, small_labels, test_size=0.2)

# Định nghĩa Dataset tùy chỉnh
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

# Hàm huấn luyện và lưu mô hình
def train_and_save_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

    # Tạo Dataset cho train và validation
    train_dataset = JobDataset(train_texts, train_labels, tokenizer)
    val_dataset = JobDataset(val_texts, val_labels, tokenizer)

    # Định nghĩa các tham số cho việc huấn luyện
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch"
    )

    # Tạo Trainer và bắt đầu huấn luyện
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    # Lưu mô hình đã huấn luyện
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

# Hàm tải mô hình đã huấn luyện
def load_model(model_path="./trained_model"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model

if __name__ == "__main__":
    # Huấn luyện và lưu mô hình
    train_and_save_model()
