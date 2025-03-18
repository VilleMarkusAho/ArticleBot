from flask import Blueprint, request, jsonify
from Models.generator import generate_text

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/generate', methods=['POST'])
def generate():
    data = request.json
    print(data)
    
    prompt = data.get("prompt", "")

    response = generate_text(prompt)
    return jsonify({"response": response}), 200