"""
Minimal Flask Fact Checking API
Returns only claims and their verification status.
"""

from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# System prompt for LLM
FACT_CHECK_PROMPT = """
Extract the factual claims from the provided text and verify each one.
Classify each claim as:
- TRUE: The claim is factually correct
- FALSE: The claim is factually incorrect
- UNVERIFIED: Cannot be verified with available information

Return ONLY a JSON array in this exact format:
[
  {
    "claim": "The exact claim text",
    "result": "TRUE/FALSE/UNVERIFIED"
  }
]

Do not include any explanations, corrections, or additional information.
"""

@app.route('/api/check', methods=['POST'])
def check_facts():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            "error": "Missing 'text' field in request"
        }), 400
    
    text = data['text']
    
    try:
        # Ask LLM to extract and verify claims
        messages = [
            SystemMessage(content=FACT_CHECK_PROMPT),
            HumanMessage(content=text)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        
        # Find and extract JSON
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            results = json.loads(json_str)
        else:
            # Try to parse the entire content as JSON
            results = json.loads(content)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to process: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)