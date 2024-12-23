from flask import Flask
from config import Config
from extensions import db, jwt

# Create and initialize the Flask app inside the function
def create_app():
    # Initialize the Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)

    # Avoid circular import by importing the routes here
    from routes import register_routes
    register_routes(app)  # Register routes with the app

    return app

# Run the app only if this file is executed directly
if __name__ == '__main__':
    create_app().run(debug=True)
