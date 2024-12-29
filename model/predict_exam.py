import torch
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification

from model.train_model import val_texts, val_labels


def predict_job_title(user_profile, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(user_profile, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label = torch.argmax(logits, dim=1).item()
    confidence = torch.max(probabilities, dim=1).values.item()
    probabilities_list = probabilities.cpu().tolist()[0]

    # In kết quả
    print("\n===== KẾT QUẢ DỰ ĐOÁN =====")
    print(f"Dự đoán nhãn: {predicted_label}")
    print(f"Độ tự tin: {confidence * 100:.2f}%")
    print("Phân phối xác suất:")
    for i, prob in enumerate(probabilities_list):
        print(f" - Nhãn {i}: {prob * 100:.2f}%")


model_path = './trained_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

user_profile_example = "Chuyên viên công nghệ sinh học"

predict_job_title(user_profile_example, model, tokenizer)

def evaluate_model(model, tokenizer, val_texts, val_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dự đoán trên tập kiểm tra
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    all_preds = []

    for text in val_texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        all_preds.append(predicted_label)

    # Tính accuracy
    accuracy = accuracy_score(val_labels, all_preds)
    return accuracy

# Sử dụng hàm để tính độ chính xác trên tập kiểm tra
accuracy = evaluate_model(model, tokenizer, val_texts, val_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")