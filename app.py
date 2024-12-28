from flask import Flask
from config import Config
from extensions import db, jwt

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    from routes import register_routes
    register_routes(app)

    return app

# Run the app
if __name__ == '__main__':
    create_app().run(debug=True)
