import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_embeddings(texts, model, tokenizer, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)

def recommend_jobs(user_profile, job_titles, model_name='bert-base-uncased', top_n=10, similarity_threshold=0.7):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    job_titles = [job for job in job_titles if job and job != 'undefined']
    user_profile_embedding = get_embeddings([user_profile], model, tokenizer)
    job_embeddings = get_embeddings(job_titles, model, tokenizer)
    similarities = cosine_similarity(user_profile_embedding.cpu().numpy(), job_embeddings.cpu().numpy()).flatten()
    relevant_jobs = [job_titles[i] for i in range(len(job_titles)) if similarities[i] >= similarity_threshold]
    if len(relevant_jobs) < top_n:
        top_10_indices = similarities.argsort()[::-1][:top_n]
        recommended_jobs = [job_titles[i] for i in top_10_indices]
    else:
        recommended_jobs = relevant_jobs

    return recommended_jobs

def load_job_titles_from_csv(file_path):
    df = pd.read_csv(file_path)

    job_titles = df['job_title'].tolist()
    return job_titles

if __name__ == "__main__":
    file_path = 'dataset_job_scraping.csv'
    job_titles = load_job_titles_from_csv(file_path)

    user_profile = "Looking for a software engineering role with expertise in Python and machine learning."

    recommended_jobs = recommend_jobs(user_profile, job_titles)
    print("Top 10 recommended jobs:")
    for i, job in enumerate(recommended_jobs, 1):
        print(f"{i}. {job}")
