from datetime import datetime
from zoneinfo import ZoneInfo

from app import db

class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    tokens = db.relationship('UserToken', back_populates='user', cascade='all, delete-orphan')

class UserToken(db.Model):
    __tablename__ = 'user_tokens'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    access_token = db.Column(db.Text, nullable=False)
    refresh_token = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")))
    access_token_expiry = db.Column(db.DateTime, nullable=False)
    refresh_token_expiry = db.Column(db.DateTime, nullable=False)
    is_blacklisted = db.Column(db.Boolean, default=False)

    user = db.relationship('User', back_populates='tokens')
