from app import app, db

# Khởi tạo cơ sở dữ liệu
with app.app_context():
    db.create_all()
