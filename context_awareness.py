from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
from flask_cors import CORS
from langchain_community.tools import TavilySearchResults

# Load environment variables
load_dotenv()

# Configure API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# Initialize Tavily Search Tool
tool = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

CONTEXT_PROMPT = """
You are a helpful assistant that provides additional context and background information for articles. 
When given an article, analyze it and generate the following:
1. A concise summary of the article.
2. Key points or highlights from the article.
3. Additional background information or context to help the reader better understand the topic.
4. Links to relevant sources or further reading (if available).

Return the response in JSON format with the following structure:
{
  "summary": "A brief summary of the article",
  "key_points": ["Key point 1", "Key point 2", "Key point 3"],
  "background_context": "Additional context or background information",
  "relevant_sources": [
    {
      "title": "Source title",
      "url": "Source URL"
    }
  ]
}
"""

def fetch_background_context(query):
    """Fetch background context for a query using Tavily Search Tool."""
    try:
        search_results = tool.invoke({"query": query})
        return search_results
    except Exception as e:
        print(f"Error fetching background context: {e}")
        return None

@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            "error": "Missing 'text' field in request"
        }), 400
    
    text = data['text']
    
    try:
        # Ask LLM to generate context and summary
        messages = [
            SystemMessage(content=CONTEXT_PROMPT),
            HumanMessage(content=text)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        
        # Parse the LLM response as JSON
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            return jsonify({
                "error": "Failed to parse LLM response as JSON"
            }), 500
        
        # Fetch additional context using Tavily
        background_context = fetch_background_context(text)
        if background_context:
            # Extract relevant sources from Tavily results
            relevant_sources = []
            for result in background_context:  # Tavily returns a list of results
                if isinstance(result, dict):  # Ensure each result is a dictionary
                    relevant_sources.append({
                        "url": result.get("url", "No URL")
                    })
            results["relevant_sources"] = relevant_sources
        else:
            results["relevant_sources"] = []
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to process: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5009))
    app.run(host='0.0.0.0', port=port, debug=True)