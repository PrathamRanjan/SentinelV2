"""
WhatsApp Fact Checking Bot powered by Twilio
Receives messages via WhatsApp and uses LLM to fact-check claims
Includes audio transcription using OpenAI's Whisper
"""
from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
import tempfile
from flask_cors import CORS
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import openai

# Load environment variables
load_dotenv()

# Configure API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# Enhanced system prompt for better fact detection and evaluation
FACT_CHECK_PROMPT = """
Analyze the text for factual claims and verify each significant claim.
Focus on scientific, medical, political, and historical claims that can be objectively evaluated.
Ignore rhetorical statements, opinions, and calls to action like "forward this message".

For each factual claim:
1. Extract the specific claim in its exact wording
2. Evaluate if it is TRUE, FALSE, or UNVERIFIED based on established knowledge
3. For FALSE claims, provide a brief one-sentence correction

Return ONLY a JSON array in this exact format:
[
  {
    "claim": "The exact claim text",
    "result": "TRUE/FALSE/UNVERIFIED",
    "correction": "Brief correction for FALSE claims only"
  }
]

Example of good claim extraction:
✓ "Drinking hot lemon water kills 99% of viruses including COVID-19"
✗ "Big Pharma doesn't want you to know this" (too vague/opinion-based)

Do not include any explanations, additional context, or formatting outside the JSON structure.
"""

def download_media(media_url):
    """
    Download media from Twilio's servers
    """
    try:
        # Get the correct media URL by removing .json if present
        correct_media_url = media_url.rstrip('.json')
        
        # Create a session with the credentials
        session = requests.Session()
        session.auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Make the request
        response = session.get(correct_media_url)
        response.raise_for_status()
        
        # Get content information
        content_type = response.headers.get('Content-Type', '')
        
        # Choose appropriate file extension
        if 'mp3' in content_type or 'mpeg' in content_type:
            ext = '.mp3'
        elif 'ogg' in content_type:
            ext = '.ogg'
        elif 'wav' in content_type:
            ext = '.wav'
        else:
            ext = '.audio'  # generic extension
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        return None
    
def transcribe_audio(audio_file_path):
    """
    Transcribe audio using OpenAI's Whisper API
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up the temporary file
        os.unlink(audio_file_path)
        
        return transcription.text
    except Exception as e:
        # Clean up the temporary file
        try:
            os.unlink(audio_file_path)
        except:
            pass
        return None

def perform_fact_check(text):
    """
    Send the text to the LLM for fact-checking
    """
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
        
        return results
    
    except Exception as e:
        return [{"claim": "Error processing request", "result": "UNVERIFIED", "correction": ""}]

def format_fact_check_results(results):
    """
    Format the fact-checking results into a readable WhatsApp message with improved styling
    """
    if not results or len(results) == 0:
        return "⚠️ No verifiable claims were identified in this message."
    
    response = "📊 *FACT CHECK RESULTS*\n\n"
    
    for item in results:
        claim = item.get("claim", "")
        result = item.get("result", "UNVERIFIED")
        correction = item.get("correction", "")
        
        if result == "TRUE":
            result_emoji = "✅ TRUE"
        elif result == "FALSE":
            result_emoji = "❌ FALSE"
        else:
            result_emoji = "❓ UNVERIFIED"
            
        response += f"*Claim:* {claim}\n*Result:* {result_emoji}\n"
        
        # Include correction for FALSE claims
        if result == "FALSE" and correction:
            response += f"*Fact:* {correction}\n"
        
        response += "\n"
    
    response += "⚠️ *Beware of messages asking you to forward to others!*\nForward more content to check its accuracy."
    return response

def detect_misinformation_patterns(text):
    """
    Detect common patterns in viral misinformation
    Returns a list of warning flags
    """
    text = text.lower()
    warnings = []
    
    # Common misinformation patterns
    if "forward this to" in text or "share with" in text or "send to" in text:
        warnings.append("⚠️ Message asks for forwarding - common in misinformation campaigns")
    
    if "big pharma" in text or "doctors don't want you to know" in text or "they don't want you to know" in text:
        warnings.append("⚠️ Claims about information suppression often lack evidence")
    
    if "miracle cure" in text or "kills 99%" in text or "cures all" in text:
        warnings.append("⚠️ Claims about miracle cures or perfect effectiveness are typically exaggerated")
    
    if "scientists have discovered" in text and not ("published in" in text or "journal" in text or "study link" in text):
        warnings.append("⚠️ Vague references to 'scientists' without specific sources")
    
    if "breaking news" in text and ("forward" in text or "share" in text):
        warnings.append("⚠️ Urgent 'breaking news' asking for shares is a red flag")
    
    return warnings

def enhance_fact_check_response(formatted_response, original_text):
    """
    Add misinformation pattern warnings to the fact check response
    """
    warnings = detect_misinformation_patterns(original_text)
    
    if warnings:
        formatted_response += "\n\n*MISINFORMATION RED FLAGS:*\n"
        for warning in warnings:
            formatted_response += f"{warning}\n"
    
    return formatted_response

def send_whatsapp_message(to_number, message_body):
    """
    Send a WhatsApp message via Twilio
    to_number should be in format 'whatsapp:+1XXXXXXXXXX'
    """
    try:
        # Make sure to properly initialize the client with each call
        # This ensures the credentials are fresh
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=to_number
        )
        return message.sid
    except Exception as e:
        return None

@app.route('/api/check', methods=['POST'])
def check_facts_api():
    """
    API endpoint for direct fact-checking
    """
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            "error": "Missing 'text' field in request"
        }), 400
    
    text = data['text']
    results = perform_fact_check(text)
    
    return jsonify(results)

@app.route('/webhook', methods=['POST'])
def twilio_webhook():
    """
    Webhook endpoint for Twilio WhatsApp messages
    """
    # Get the message content
    incoming_msg = request.values.get('Body', '').strip()
    sender = request.values.get('From', '')
    
    # Create a response object
    resp = MessagingResponse()
    
    # Check if this is a join or help message
    if incoming_msg:
        lower_msg = incoming_msg.lower()
        if lower_msg in ["join", "hello", "hi", "hey", "start"]:
            resp.message("👋 Welcome to FactCheck Bot! Send me any message, news article, or claim, and I'll check its factual accuracy for you. You can also send voice messages for fact-checking.")
            return str(resp)
    
    # Check for media content
    num_media = int(request.values.get('NumMedia', 0))
    
    # Handle media messages
    if num_media > 0:
        media_url = request.values.get('MediaUrl0', '')
        media_content_type = request.values.get('MediaContentType0', '')
        
        if media_content_type.startswith('audio/') or media_content_type.startswith('voice/'):
            try:
                # Download and process audio
                file_path = download_media(media_url)
                if file_path:
                    # Transcribe audio
                    transcribed_text = transcribe_audio(file_path)
                    
                    if transcribed_text:
                        # Process the transcription
                        results = perform_fact_check(transcribed_text)
                        formatted_response = format_fact_check_results(results)
                        enhanced_response = enhance_fact_check_response(formatted_response, transcribed_text)
                        
                        # Send combined response with transcription and fact check
                        full_response = f"📝 *Transcription:*\n\n{transcribed_text}\n\n{enhanced_response}"
                        resp.message(full_response)
                    else:
                        resp.message("I couldn't transcribe this audio. Please try sending clearer audio or text directly.")
                else:
                    resp.message("I had trouble accessing this audio file. Please try again or send text directly.")
            except Exception:
                resp.message("I encountered an error processing your audio. Please try sending text instead.")
        
        elif media_content_type.startswith('image/') or media_content_type.startswith('video/'):
            # Skip image and video processing
            resp.message("I currently only fact-check text and audio messages. Please send me text or voice messages to verify.")
        
        else:
            resp.message("I received your media but can't process this type. I can fact-check text and audio messages.")
    
    # Handle text messages
    elif incoming_msg:
        # Process text message with enhanced fact checking
        try:
            results = perform_fact_check(incoming_msg)
            formatted_response = format_fact_check_results(results)
            
            # Enhance response with misinformation pattern detection
            enhanced_response = enhance_fact_check_response(formatted_response, incoming_msg)
            
            resp.message(enhanced_response)
        except Exception:
            resp.message("I encountered an error while fact-checking. Please try again with a different message.")
    
    # Empty message
    else:
        resp.message("Please send me a message to fact-check.")
    
    return str(resp)

@app.route('/send-welcome', methods=['POST'])
def send_welcome_message():
    """
    API endpoint to send a welcome message to a user
    """
    data = request.json
    
    if not data or 'to' not in data:
        return jsonify({
            "error": "Missing 'to' field in request"
        }), 400
    
    to_number = data['to']
    # Ensure the to_number has the WhatsApp prefix
    if not to_number.startswith('whatsapp:'):
        to_number = f"whatsapp:{to_number}"
    
    welcome_message = "Hello! I'm WhatsappBot - your personalized fact checker! Send me any text, news, voice messages, or claims to verify."
    message_sid = send_whatsapp_message(to_number, welcome_message)
    
    if message_sid:
        return jsonify({
            "success": True,
            "message_sid": message_sid
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to send message"
        }), 500

if __name__ == '__main__':
    # Send a notification on startup
    print("WhatsApp Fact Checker Bot starting up...")
    
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)