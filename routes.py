from flask import render_template, redirect, url_for
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
import pandas as pd
import os
import torch
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, UserToken
import jwt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import redis
dataset_path = 'dataset_job_scraping.csv'
columns = ['No', 'Job_title', 'Job_Company', 'Location', 'Status', 'Time Period']
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Kiểm tra nếu file tồn tại
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

# Kiểm tra thông tin đăng nhập
def check_login_credentials(username, password):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        return user
    return None

# Hàm giải mã JWT
def decode_token(token):
    try:
        secret_key = '32313312323ww'  # Đảm bảo rằng bạn sử dụng key chính xác
        decoded = jwt.decode(token, secret_key, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
def get_embeddings(texts, model, tokenizer):
    # Tokenize các văn bản đầu vào
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    # Lấy embeddings từ token [CLS]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Sử dụng embedding của token [CLS] (token đầu tiên)
    return embeddings

def register_routes(app):

    # Trang chủ (có hoặc không có xác thực)
    @app.route('/')
    @jwt_required(optional=True)
    def home():
        current_user = get_jwt_identity()
        if current_user:
            return render_template('index.html', user=current_user)
        else:
            return redirect(url_for('login_page'))

    # Đăng ký người dùng mới
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

    # Đăng nhập
    # Đăng nhập
    @app.route('/login', methods=['POST'])
    def login():
        username = request.json.get('username')
        password = request.json.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        user = check_login_credentials(username, password)
        if user:  # Nếu người dùng hợp lệ
            # Tạo access token
            access_token = create_access_token(identity=username)
            return jsonify({'access_token': access_token}), 200  # Trả về token
        else:
            return jsonify({'error': 'Thông tin đăng nhập không đúng.'}), 401


    # Gợi ý công việc
    # @app.route('/recommend_jobs', methods=['GET', 'POST'])
    # def recommend_jobs():
    #     try:
    #         app.logger.info("Endpoint /recommend_jobs accessed")
    #         if request.method == 'GET':
    #             return render_template('index.html')
    #
    #         # Kiểm tra dữ liệu công việc
    #         if df.empty:
    #             app.logger.error("Job data (df) is empty")
    #             return jsonify({"error": "No job data available"}), 400
    #         app.logger.info(f"DataFrame loaded successfully with shape: {df.shape}")
    #
    #         # Lấy hồ sơ người dùng từ request
    #         user_profile = request.json.get('user_profile', '').strip()
    #         if not user_profile:
    #             app.logger.error("User profile is missing")
    #             return jsonify({"error": "User profile is required"}), 400
    #         app.logger.info(f"User profile received: {user_profile}")
    #
    #         # Kiểm tra dữ liệu cột 'Job_title'
    #         if df['Job_title'].isnull().any():
    #             app.logger.error("Null values found in Job_title column")
    #             return jsonify({"error": "Invalid job titles in dataset"}), 400
    #         app.logger.info("Job titles verified successfully")
    #
    #         # Vector hóa tiêu đề công việc và hồ sơ người dùng
    #         vectorizer = TfidfVectorizer()
    #         job_vectors = vectorizer.fit_transform(df['Job_title'])
    #         user_vector = vectorizer.transform([user_profile])
    #         app.logger.info(f"Vectorization successful: job_vectors={job_vectors.shape}, user_vector={user_vector.shape}")
    #
    #         # Tính độ tương đồng cosine
    #         similarities = cosine_similarity(user_vector, job_vectors).flatten()
    #         app.logger.info(f"Cosine similarities calculated: {similarities[:5]}")
    #
    #         # # Thêm cột độ tương đồng vào DataFrame
    #         df['Similarity'] = similarities
    #         # app.logger.info(f"Added Similarity column to DataFrame")
    #
    #         # Sắp xếp công việc theo độ tương đồng giảm dần
    #         sorted_jobs = df.sort_values(by='Similarity', ascending=False)
    #         app.logger.info(f"Sorted jobs based on similarity")
    #
    #         # Loại bỏ các công việc trùng lặp
    #         unique_jobs = sorted_jobs.drop_duplicates(subset=['Job_title', 'Job_Company'])
    #         app.logger.info(f"Removed duplicate jobs. Unique jobs count: {len(unique_jobs)}")
    #
    #         # Chỉ lấy 20 công việc phù hợp nhất
    #         recommended_jobs = unique_jobs.head(10).to_dict(orient='records')
    #         app.logger.info(f"Recommended jobs: {len(recommended_jobs)}")
    #
    #         # Trả về danh sách gợi ý
    #         return jsonify({"recommended_jobs": recommended_jobs}), 200
    #
    #     except Exception as e:
    #         app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
    #         return jsonify({"error": "An error occurred while fetching the job recommendations"}), 500


    @app.route('/recommend_jobs', methods=['GET', 'POST'])
    def recommend_jobs():
        try:
            app.logger.info("Endpoint /recommend_jobs accessed")

            if request.method == 'GET':
                return render_template('index.html')

            if df.empty:
                app.logger.error("Job data (df) is empty")
                return jsonify({"error": "No job data available"}), 400

            app.logger.info(f"DataFrame loaded successfully with shape: {df.shape}")

            user_profile = request.json.get('user_profile', '').strip()
            if not user_profile:
                app.logger.error("User profile is missing")
                return jsonify({"error": "User profile is required"}), 400

            app.logger.info(f"User profile received: {user_profile}")

            if df['Job_title'].isnull().any():
                app.logger.error("Null values found in Job_title column")
                return jsonify({"error": "Invalid job titles in dataset"}), 400

            app.logger.info("Job titles verified successfully")

            vectorizer = TfidfVectorizer()
            job_vectors = vectorizer.fit_transform(df['Job_title'])
            user_vector = vectorizer.transform([user_profile])

            app.logger.info(f"Vectorization successful: job_vectors={job_vectors.shape}, user_vector={user_vector.shape}")

            similarities = cosine_similarity(user_vector, job_vectors).flatten()
            app.logger.info(f"Cosine similarities calculated: {similarities[:5]}")

            df['Similarity'] = similarities
            sorted_jobs = df.sort_values(by='Similarity', ascending=False)

            unique_jobs = sorted_jobs.drop_duplicates(subset=['Job_title', 'Job_Company'])
            num_jobs = min(50, len(unique_jobs))  # Nếu có ít hơn 30 công việc, lấy tất cả
            recommended_jobs = unique_jobs.head(num_jobs).to_dict(orient='records')

            app.logger.info(f"Recommended jobs: {len(recommended_jobs)}")

            redis_client.lpush('job_search_history', user_profile)  # Lưu hồ sơ người dùng vào Redis

            # Trả về kết quả gợi ý
            return jsonify({"recommended_jobs": recommended_jobs}), 200

        except Exception as e:
            app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred while fetching the job recommendations"}), 500
    @app.route('/get_search_history', methods=['GET'])
    def get_search_history():
        try:
            # Lấy toàn bộ lịch sử tìm kiếm từ Redis
            history = redis_client.lrange('job_search_history', 0, -1)

            # Trả về lịch sử dưới dạng JSON
            return jsonify({"history": history}), 200
        except Exception as e:
            app.logger.error(f"Error occurred: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred while fetching search history"}), 500
    # Trang login
    @app.route('/login_page', methods=['GET', 'POST'])
    def login_page():
        return render_template('login.html')
    @app.route('/logout', methods=['POST'])
    def logout():
        try:
            current_user = get_jwt_identity()

            token = request.headers.get('Authorization').split(" ")[1]
            redis_client.sadd('blacklist', token)

            app.logger.info(f"User {current_user} đã đăng xuất thành công.")
            return jsonify({"message": "Log out successfully!"}), 200
        except Exception as e:
            app.logger.error(f"Lỗi trong khi đăng xuất: {str(e)}", exc_info=True)
            return jsonify({"error": "An error occurred while fetching log out."}), 500

