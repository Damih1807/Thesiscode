from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import render_template, redirect, url_for
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
import pandas as pd
import os
import torch
from transformers import BertTokenizer
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, UserToken
import jwt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import redis
from model import train_model
dataset_path = 'dataset_job_scraping.csv'
columns = ['No', 'Job_title', 'Job_Company', 'Location', 'Status', 'Time Period']
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = train_model.load_model()
model.eval()
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    df.columns = columns
    df = df.drop(['No', 'Time Period', 'Status'], axis=1)
else:
    df = pd.DataFrame(columns=['Job_title', 'Job_Company', 'Location'])

# Khởi tạo vectorizer
vectorizer = TfidfVectorizer()

# Tạo job vectors từ job titles
def create_job_vectors():
    return vectorizer.fit_transform(df['Job_title']) if not df.empty else None

job_vectors = create_job_vectors()

def check_login_credentials(username, password):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        return user
    return None

def decode_token(token):
    try:
        secret_key = '32313312323ww'
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
def get_text_embedding(text, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        # Lấy embedding từ lớp đầu ra cuối cùng hoặc lớp pooler
        return outputs.logits[0].numpy()

def register_routes(app):

    # Main
    @app.route('/')
    @jwt_required(optional=True)
    def home():
        current_user = get_jwt_identity()
        if current_user:
            return render_template('index.html', user=current_user)
        else:
            return redirect(url_for('login_page'))

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            data = request.form
            username = data.get('username')
            password = data.get('password')
            confirm_password = data.get('confirmPassword')

            if not username or not password or not confirm_password:
                return jsonify({'error': 'Username, password, and confirmPassword are required'}), 400
            if password != confirm_password:
                return jsonify({'error': 'Passwords do not match'}), 400

            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                return jsonify({'error': 'Username already exists'}), 409

            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password)

            try:
                db.session.add(new_user)
                db.session.commit()
                return redirect(url_for('login_page'))
            except Exception as e:
                db.session.rollback()
                return jsonify({'error': 'Registration failed', 'details': str(e)}), 500

        return render_template('register.html')

    # @app.route('/login', methods=['POST'])
    # def login():
    #     username = request.json.get('username')
    #     password = request.json.get('password')
    #
    #     if not username or not password:
    #         return jsonify({'error': 'Username and password are required'}), 400
    #
    #     user = check_login_credentials(username, password)
    #     if user:
    #         access_token = create_access_token(identity=username)
    #         return jsonify({'access_token': access_token}), 200  # Trả về token
    #     else:
    #         return jsonify({'error': 'Thông tin đăng nhập không đúng.'}), 401

    @app.route('/recommend_jobs', methods=['GET', 'POST'])
    def recommend_jobs():
        try:
            if request.method == 'GET':
                return render_template('index.html')

            if df.empty or df['Job_title'].isnull().any():
                return jsonify({"error": "Job dataset is invalid or empty"}), 400

            user_profile = request.json.get('user_profile', '').strip()
            if not user_profile:
                return jsonify({"error": "User profile is required"}), 400

            job_vectors = vectorizer.fit_transform(df['Job_title'])
            user_vector = vectorizer.transform([user_profile])

            similarities = cosine_similarity(user_vector, job_vectors).flatten()
            df['Similarity'] = similarities
            sorted_jobs = df.sort_values(by='Similarity', ascending=False).drop_duplicates(subset=['Job_title', 'Job_Company'])

            recommended_jobs = sorted_jobs.head(10).to_dict(orient='records')

            try:
                redis_client.lpush('job_search_history', user_profile)
            except redis.exceptions.RedisError as e:
                app.logger.error(f"Error saving to Redis: {str(e)}")

            return jsonify({"recommended_jobs": recommended_jobs}), 200
        except Exception as e:
            app.logger.error(f"Error in recommend_jobs: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred"}), 500
    @app.route('/login', methods=['POST'])
    def login():
        username = request.json.get('username')
        password = request.json.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        user = check_login_credentials(username, password)
        if user:
            # Tạo access token (và refresh token nếu cần)
            access_token = create_access_token(identity=username)

            # Ví dụ về thời gian hết hạn
            access_token_expiry = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")) + timedelta(minutes=15)
            refresh_token_expiry = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")) + timedelta(days=30)

            # Nếu bạn muốn tạo refresh token (tùy thuộc vào cách cài đặt của bạn)
            refresh_token = create_access_token(identity=username, expires_delta=timedelta(days=30))

            # Lưu token vào bảng UserToken
            new_user_token = UserToken(
                user_id=user.user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                access_token_expiry=access_token_expiry,
                refresh_token_expiry=refresh_token_expiry,
                is_blacklisted=False
            )
            try:
                db.session.add(new_user_token)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                return jsonify({'error': 'Lưu token thất bại', 'details': str(e)}), 500

            return jsonify({'access_token': access_token}), 200  # Trả về token cho client
        else:
            return jsonify({'error': 'Thông tin đăng nhập không đúng.'}), 401

    @app.route('/logout', methods=['POST'])
    @jwt_required()
    def logout():
        try:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Invalid or missing authorization header"}), 400

            token = auth_header.split(" ")[1]
            redis_client.sadd('blacklist', token)

            return jsonify({"message": "Logged out successfully!"}), 200
        except Exception as e:
            app.logger.error(f"Error in logout: {str(e)}", exc_info=True)
            return jsonify({"error": "Failed to log out"}), 500

    @app.route('/login_page', methods=['GET', 'POST'])
    def login_page():
        return render_template('login.html')
    @app.route('/get_search_history', methods=['GET'])
    def get_search_history():
        try:
            history = redis_client.lrange('job_search_history', 0, -1)

            return jsonify({"history": history}), 200
        except Exception as e:
            app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred while fetching search history"}), 500