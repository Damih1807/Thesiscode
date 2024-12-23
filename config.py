import os
from datetime import timedelta

class Config:
    # Cấu hình chung
    SECRET_KEY = os.getenv('SECRET_KEY', '32313312323ww')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', '32313312323ww')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://postgres:postgres@localhost/Web Jobs')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Cấu hình môi trường
    ENV = 'development'  # 'production'
    DEBUG = True

    # Cấu hình JWT
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)  # 15 phút
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)  # 7 ngày
    JWT_TOKEN_LOCATION = ['headers']  # Hoặc ['cookies']
    JWT_COOKIE_SECURE = False  # Bật True nếu dùng HTTPS

    # Cấu hình Logging
    LOGGING_LEVEL = 'INFO'
