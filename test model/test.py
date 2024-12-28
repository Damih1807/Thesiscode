import torch
from torch import device
from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Đường dẫn tới dataset
dataset_path = 'dataset_job_scraping.csv'

# Hàm lấy embeddings từ văn bản sử dụng mô hình BERT
def evaluate_recommendation_system(user_profiles, job_titles, true_labels, model_name='bert-base-uncased', top_n=10, similarity_threshold=0.7, batch_size=16):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    job_titles = [job for job in job_titles if job and job != 'undefined']
    all_preds = []

    # Tokenize user profiles trước
    user_profiles_tokens = tokenizer(user_profiles, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids_user = user_profiles_tokens['input_ids'].to(device)
    attention_mask_user = user_profiles_tokens['attention_mask'].to(device)

    # Lặp qua các job_titles theo batch_size
    for user_profile, true_label in zip(user_profiles, true_labels):
        with torch.no_grad():
            user_profile_embedding = model(input_ids_user, attention_mask=attention_mask_user).last_hidden_state[:, 0, :]

        # Xử lý job titles theo batch
        job_embeddings = []
        for i in range(0, len(job_titles), batch_size):
            batch_job_titles = job_titles[i:i+batch_size]
            job_tokens = tokenizer(batch_job_titles, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids_jobs = job_tokens['input_ids'].to(device)
            attention_mask_jobs = job_tokens['attention_mask'].to(device)

            with torch.no_grad():
                batch_job_embeddings = model(input_ids_jobs, attention_mask=attention_mask_jobs).last_hidden_state[:, 0, :]
            job_embeddings.append(batch_job_embeddings)

        # Kết hợp embeddings của tất cả các batch
        job_embeddings = torch.cat(job_embeddings, dim=0)

        # Tính cosine similarity giữa user profile và các job titles
        similarities = cosine_similarity(user_profile_embedding.cpu().numpy(), job_embeddings.cpu().numpy()).flatten()

        # Chọn top_n job titles có độ tương đồng cao nhất
        top_indices = similarities.argsort()[::-1][:top_n]
        recommended_jobs = [job_titles[i] for i in top_indices]

        # Kiểm tra nếu true_label có trong danh sách các job titles được đề xuất
        all_preds.append(1 if true_label in recommended_jobs else 0)

    # Tính các chỉ số đánh giá
    accuracy = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds)
    recall = recall_score(true_labels, all_preds)
    f1 = f1_score(true_labels, all_preds)

    return accuracy, precision, recall, f1


# Hàm tải danh sách job title từ file CSV
def load_job_titles_from_csv(file_path):
    column_names = ['No', 'Job_title', 'Job_Company', 'Location', 'Status', 'Time Period']
    # Đọc file CSV và gán tên cột nếu trong file không có sẵn
    df = pd.read_csv(file_path, names=column_names, header=None)

    # Kiểm tra sự tồn tại của cột 'Job_title'
    if 'Job_title' in df.columns:
        job_titles = df['Job_title'].tolist()
    else:
        print("Cột 'Job_title' không tồn tại trong file CSV.")
        job_titles = []

    return job_titles

if __name__ == "__main__":
    # Đọc danh sách job title từ file CSV
    file_path = dataset_path
    job_titles = load_job_titles_from_csv(file_path)

    # Kiểm tra nếu job titles không rỗng
    if not job_titles:
        print("Không có job titles trong file.")
    else:
        print(f"Job Titles: {job_titles[:5]}")  # In ra 5 công việc đầu tiên

    # Dữ liệu ví dụ về user profiles và true labels (dùng cho đánh giá)
    user_profiles = [
        "Looking for a software engineering role with expertise in Python and machine learning.",
        "Looking for a marketing position with strong communication skills."
    ]
    true_labels = [
        "Software Engineer",
        "Marketing Manager"
    ]

    # Đánh giá hệ thống gợi ý việc làm
    accuracy, precision, recall, f1 = evaluate_recommendation_system(user_profiles, job_titles, true_labels)

    # In ra các chỉ số đánh giá
    print(f"Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
