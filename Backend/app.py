from flask import Flask
from Api.routes import chatbot_bp

app = Flask(__name__)
app.register_blueprint(chatbot_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)
