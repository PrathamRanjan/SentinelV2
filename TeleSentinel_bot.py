"""
Telegram Fact Checking Bot (Webhook Only)
Uses webhooks exclusively for Telegram integration
Improved with WhatsApp bot's fact-checking logic
"""
from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
import tempfile
import re
import traceback
import telebot
from telebot import types
import openai
import time
import threading

# Load environment variables
load_dotenv()

# Configure API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7922034590:AAEgoL0RYMYERex5dx94Im8lDuD9KzPKOnU")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Initialize Telegram bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# System prompt for extracting claims from text
EXTRACT_CLAIMS_PROMPT = """
Analyze the provided text and extract 3-4 specific factual claims that can be verified.

For each claim:
1. Extract the exact statement from the text that can be verified as true or false
2. Make sure these are substantive factual claims, not opinions or subjective statements
3. Focus on claims that would be important for readers to know the accuracy of
4. Include sufficient context to make the claim clear

Return ONLY a JSON array in this exact format:
[
  {
    "claim": "The exact claim from the text with necessary context",
    "search_query": "Suggested search terms to verify this claim"
  }
]

Do not attempt to verify the claims yourself. Just identify and contextualize them for verification.
"""

# System prompt for verifying with Serper search results
SERPER_VERIFICATION_PROMPT = """
You are a fact-checker with a reputation for accuracy and attention to detail.

Fact-check the following claim using the search results provided:

Claim: {{claim}}

Search Results: {{search_results}}

Based on these search results, determine if the claim is TRUE, FALSE, or UNVERIFIED.

Your response must be in this exact JSON format:
{
  "claim": "{{claim}}",
  "result": "TRUE/FALSE/UNVERIFIED",
  "summary": "A one-sentence summary of your verdict",
  "detailed_analysis": "A detailed explanation of your reasoning (2-3 sentences)",
  "sources": [
    {
      "name": "Website or Publication Name",
      "url": "Source URL"
    }
  ]
}

Guidelines:
- Only mark a claim as TRUE if credible sources clearly support it
- Only mark a claim as FALSE if credible sources clearly refute it
- Mark as UNVERIFIED if the sources are contradictory or insufficient
- Focus on the most authoritative sources (educational institutions, scientific publications, etc.)
"""

# Functions for media handling

def download_telegram_file(file_id):
    """
    Download file from Telegram's servers and ensure it's in a compatible format
    """
    try:
        print(f"[DOWNLOAD] Starting download of file with ID: {file_id}")
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        
        print(f"[DOWNLOAD] File URL: {file_url}")
        response = requests.get(file_url)
        response.raise_for_status()
        print(f"[DOWNLOAD] File downloaded, size: {len(response.content)} bytes")
        
        # Telegram voice messages are usually in OGG format, which may need conversion
        # Always save with .mp3 extension for compatibility with Whisper API
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_file.write(response.content)
        temp_file_path = temp_file.name
        temp_file.close()
        
        print(f"[DOWNLOAD] Saved to temporary file: {temp_file_path}")
        
        # Verify the file is a valid audio file
        try:
            file_size = os.path.getsize(temp_file_path)
            print(f"[DOWNLOAD] File size verification: {file_size} bytes")
            if file_size < 100:  # Extremely small files are likely invalid
                print("[DOWNLOAD] Warning: File is too small, might be invalid")
                return None
        except Exception as e:
            print(f"[DOWNLOAD] Error verifying file: {e}")
            return None
        
        return temp_file_path
    except Exception as e:
        print(f"[DOWNLOAD] Error downloading media: {e}")
        traceback.print_exc()
        return None

# Also modify the handle_audio function to better handle Telegram's voice messages:

@bot.message_handler(content_types=['voice', 'audio'])
def handle_audio(message):
    chat_id = message.chat.id
    
    # Send a "processing" message
    processing_msg = bot.send_message(chat_id, "🔍 I'm processing your audio. This may take a moment...")
    
    try:
        # Get file ID and additional info
        if message.content_type == 'voice':
            file_id = message.voice.file_id
            duration = message.voice.duration
            print(f"[AUDIO] Processing voice message with file_id: {file_id}, duration: {duration}s")
        else:  # 'audio'
            file_id = message.audio.file_id
            duration = message.audio.duration
            print(f"[AUDIO] Processing audio file with file_id: {file_id}, duration: {duration}s")
        
        # Download the file
        file_path = download_telegram_file(file_id)
        print(f"[AUDIO] Download result: {'Success' if file_path else 'Failed'}")
        
        if file_path:
            # Verify file exists and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"[AUDIO] File validation passed. Size: {os.path.getsize(file_path)} bytes")
                
                # Transcribe the audio
                transcribed_text = transcribe_audio(file_path)
                print(f"[AUDIO] Transcription: {'Success' if transcribed_text else 'Failed'}")
                
                if transcribed_text:
                    print(f"[AUDIO] Transcribed text: {transcribed_text[:100]}...")
                    
                    # Process the transcription with fact checking
                    results = perform_serper_fact_check(transcribed_text)
                    print(f"[AUDIO] Fact check complete, got {len(results)} results")
                    
                    # Format the results (without showing transcription)
                    formatted_response = format_serper_fact_check_results(results)
                    enhanced_response = enhance_fact_check_response(formatted_response, transcribed_text)
                    
                    print(f"[AUDIO] Enhanced response length: {len(enhanced_response)}")
                    
                    # Split into chunks if too long
                    if len(enhanced_response) > 4000:
                        chunks = [enhanced_response[i:i+4000] for i in range(0, len(enhanced_response), 4000)]
                        for chunk in chunks:
                            bot.send_message(chat_id, chunk, parse_mode="Markdown")
                    else:
                        bot.send_message(chat_id, enhanced_response, parse_mode="Markdown")
                else:
                    bot.send_message(chat_id, "I couldn't transcribe this audio. Please try sending clearer audio or text directly.")
            else:
                print(f"[AUDIO] File validation failed. File exists: {os.path.exists(file_path)}")
                bot.send_message(chat_id, "The audio file appears to be invalid. Please try sending a different audio message.")
        else:
            bot.send_message(chat_id, "I had trouble accessing this audio file. Please try again or send text directly.")
    
    except Exception as e:
        print(f"[AUDIO] Error processing audio: {e}")
        traceback.print_exc()
        bot.send_message(chat_id, "I encountered an error processing your audio. Please try sending text instead.")
    
    # Delete the processing message
    bot.delete_message(chat_id, processing_msg.message_id)
def transcribe_audio(audio_file_path):
    """
    Transcribe audio using OpenAI's Whisper API
    """
    print(f"[TRANSCRIBE] Starting transcription of file: {audio_file_path}")
    try:
        print(f"[TRANSCRIBE] File exists: {os.path.exists(audio_file_path)}")
        print(f"[TRANSCRIBE] File size: {os.path.getsize(audio_file_path)} bytes")
        
        with open(audio_file_path, "rb") as audio_file:
            print("[TRANSCRIBE] File opened successfully, sending to Whisper API")
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Clean up the temporary file
        os.unlink(audio_file_path)
        print("[TRANSCRIBE] Temp file deleted")
        
        if transcription and hasattr(transcription, 'text'):
            print(f"[TRANSCRIBE] Transcription successful, text length: {len(transcription.text)}")
            print(f"[TRANSCRIBE] Transcription: {transcription.text[:100]}...")
            return transcription.text
        else:
            print("[TRANSCRIBE] Transcription object invalid or missing text property")
            return None
    except Exception as e:
        print(f"[TRANSCRIBE] Error transcribing audio: {str(e)}")
        traceback.print_exc()
        # Clean up the temporary file
        try:
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
                print("[TRANSCRIBE] Temp file deleted after error")
        except:
            pass
        return None

# Helper function for JSON processing
def fix_broken_json(json_str):
    """Fix common JSON errors from LLM outputs"""
    try:
        # Test if it's valid to begin with
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # Fix missing commas between properties
        fixed_str = re.sub(r'(".*?":\s*".*?")\s*\n\s*(".*?")',
                           r'\1,\n  \2', json_str)
        
        # Handle properties with non-string values
        fixed_str = re.sub(r'(".*?":\s*[^",\s{[].*?[^,\s{[])(\s*\n\s*)(".*?")',
                           r'\1,\2\3', fixed_str)
        
        try:
            # Test if our fix worked
            json.loads(fixed_str)
            return fixed_str
        except json.JSONDecodeError:
            # If all else fails, return the original
            return json_str
    except Exception:
        # Any other exceptions, just return the original
        return json_str

# Serper fact-checking functions - Copied from WhatsApp bot
def extract_claims(text):
    """Extract factual claims from text using LLM"""
    try:
        # For very short text, treat it as a single claim
        if len(text.split()) < 15:
            return [{
                "claim": text,
                "search_query": f"fact check {text}"
            }]
        
        messages = [
            SystemMessage(content=EXTRACT_CLAIMS_PROMPT),
            HumanMessage(content=text)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        
        # Extract JSON
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            claims = json.loads(json_str)
            return claims
        else:
            # If no proper JSON found, create a fallback claim
            return [{
                "claim": text[:200] + ("..." if len(text) > 200 else ""),
                "search_query": f"fact check {text[:100]}"
            }]
    except Exception as e:
        print(f"Error extracting claims: {e}")
        # Create a fallback claim with the first sentence
        first_sentence = text.split('.')[0] if '.' in text else text[:100]
        return [{
            "claim": first_sentence,
            "search_query": f"fact check {first_sentence}"
        }]

# Using the WhatsApp bot's search function with hardcoded API key for testing
def search_with_serper(query):
    """Search for a query using Serper API"""
    try:
        url = "https://google.serper.dev/search"
        api_key = "2509c7486d24cbd0f8facbcf4999e9d41094eed4"  # Hardcoded for testing
        
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': 5
        }
        
        print(f"[SEARCH] Using direct API key: {api_key[:4]}...{api_key[-4:]}")
        print(f"[SEARCH] Query: {query}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"[SEARCH] Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SEARCH] Found {len(result.get('organic', []))} organic results")
            return result
        else:
            print(f"[SEARCH] Full response text: {response.text}")
            return None
    except Exception as e:
        print(f"[SEARCH] Error in Serper search: {str(e)}")
        traceback.print_exc()
        return None

# Using the WhatsApp bot's verification function
def verify_with_serper_and_llama(claim_data):
    """Verify a claim using Serper search results and LLM"""
    try:
        # Extract claim text
        if isinstance(claim_data, dict):
            claim = claim_data.get("claim", "")
            search_query = claim_data.get("search_query", "")
        else:
            claim = claim_data
            search_query = ""
        
        print(f"[VERIFY] Verifying claim: {claim}")
        print(f"[VERIFY] Search query: {search_query}")
        
        # Generate a search query if not provided
        if not search_query:
            search_query = f"fact check {claim}"
        
        # Search with Serper
        search_results = search_with_serper(search_query)
        
        if not search_results:
            print("[VERIFY] No search results returned from Serper")
            return {
                "claim": claim,
                "result": "UNVERIFIED",
                "summary": "Insufficient evidence available to verify this claim.",
                "detailed_analysis": "No reliable sources were found to verify this specific claim.",
                "sources": []
            }
        
        if "organic" not in search_results or len(search_results["organic"]) == 0:
            print("[VERIFY] No organic search results in Serper response")
            return {
                "claim": claim,
                "result": "UNVERIFIED",
                "summary": "Insufficient evidence available to verify this claim.",
                "detailed_analysis": "No reliable sources were found to verify this specific claim.",
                "sources": []
            }
        
        print(f"[VERIFY] Found {len(search_results['organic'])} organic search results")
        
        # Format search results for the prompt
        formatted_results = json.dumps(search_results["organic"][:5], indent=2)
        
        # Create verification prompt
        verification_prompt = SERPER_VERIFICATION_PROMPT.replace("{{claim}}", claim).replace("{{search_results}}", formatted_results)
        
        print("[VERIFY] Sending to LLM for verification...")
        messages = [
            SystemMessage(content=verification_prompt)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        print(f"[VERIFY] LLM response first 100 chars: {content[:100]}...")
        
        # Extract JSON
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                print(f"[VERIFY] Extracted JSON string first 100 chars: {json_str[:100]}...")
                
                # Fix potential JSON formatting errors
                fixed_json_str = fix_broken_json(json_str)
                
                try:
                    result = json.loads(fixed_json_str)
                    print(f"[VERIFY] Parsed JSON result with keys: {result.keys()}")
                    
                    # Ensure result includes the claim
                    if 'claim' not in result:
                        result['claim'] = claim
                    
                    # Format sources for clean display
                    formatted_sources = []
                    if 'sources' in result and result['sources']:
                        for source in result['sources']:
                            if isinstance(source, dict) and 'name' in source and 'url' in source:
                                formatted_sources.append({
                                    'name': source['name'],
                                    'url': source['url']
                                })
                    
                    if formatted_sources:
                        result['sources'] = formatted_sources
                    else:
                        result['sources'] = []
                    
                    print(f"[VERIFY] Final verification result: {result['result']}")
                    return result
                except json.JSONDecodeError as e:
                    print(f"[VERIFY] JSON decode error: {e}")
                    print(f"[VERIFY] Problem JSON string: {fixed_json_str}")
                    raise ValueError("JSON parsing error")
            else:
                print("[VERIFY] Could not find JSON in LLM response")
                print(f"[VERIFY] Full LLM response: {content}")
                raise ValueError("JSON not found in response")
        
        except Exception as e:
            print(f"[VERIFY] Error processing verification response: {e}")
            traceback.print_exc()
            
            # Create a fallback result
            return {
                "claim": claim,
                "result": "UNVERIFIED",
                "summary": "Technical issues prevented proper verification.",
                "detailed_analysis": "Technical difficulties prevented the proper analysis of this claim.",
                "sources": []
            }
    
    except Exception as e:
        print(f"[VERIFY] Error in Serper verification: {e}")
        traceback.print_exc()
        
        # Create a fallback result
        return {
            "claim": claim_data.get("claim", claim_data) if isinstance(claim_data, dict) else claim_data,
            "result": "UNVERIFIED",
            "summary": "Technical difficulties interrupted the verification process.",
            "detailed_analysis": f"An error occurred during verification: {str(e)}",
            "sources": []
        }

def perform_serper_fact_check(text):
    """
    Main function to perform fact-checking with Serper
    """
    try:
        print(f"[FACT-CHECK] Starting fact check for text: {text[:100]}...")
        
        # Extract claims from text
        claims = extract_claims(text)
        print(f"[FACT-CHECK] Extracted {len(claims)} claims")
        
        if not claims:
            print("[FACT-CHECK] No claims found")
            return [{"claim": "No verifiable claims found in this message.", 
                     "result": "UNVERIFIED", 
                     "summary": "The message doesn't contain specific factual statements that can be verified."}]
        
        # Verify each claim (limit to 3 to keep responses manageable)
        verified_claims = []
        for i, claim_obj in enumerate(claims[:3]):
            print(f"[FACT-CHECK] Verifying claim {i+1}: {claim_obj.get('claim', '')[:100]}...")
            verification = verify_with_serper_and_llama(claim_obj)
            verified_claims.append(verification)
            print(f"[FACT-CHECK] Claim {i+1} result: {verification.get('result', 'UNKNOWN')}")
        
        return verified_claims
    
    except Exception as e:
        print(f"[FACT-CHECK] Error in Serper fact-checking: {e}")
        traceback.print_exc()
        return [{"claim": "Error processing request", 
                 "result": "UNVERIFIED", 
                 "summary": f"Technical error during fact-checking: {str(e)}"}]

def format_serper_fact_check_results(results):
    """
    Format the Serper fact-checking results into a readable message
    """
    if not results or len(results) == 0:
        return "⚠️ No verifiable claims were identified in this message."
    
    response = "📊 *FACT CHECK RESULTS*\n\n"
    
    for item in results:
        claim = item.get("claim", "")
        result = item.get("result", "UNVERIFIED")
        summary = item.get("summary", "")
        detailed_analysis = item.get("detailed_analysis", "")
        sources = item.get("sources", [])
        
        if result == "TRUE":
            result_emoji = "✅ TRUE"
        elif result == "FALSE":
            result_emoji = "❌ FALSE"
        else:
            result_emoji = "❓ UNVERIFIED"
            
        response += f"*Claim:* {claim}\n*Result:* {result_emoji}\n"
        
        if summary:
            response += f"*Summary:* {summary}\n"
        
        if detailed_analysis:
            response += f"*Analysis:* {detailed_analysis}\n"
        
        # Add sources if available
        if sources:
            response += "*Sources:*\n"
            for source in sources[:2]:  # Limit to first 2 sources
                if isinstance(source, dict):
                    name = source.get("name", "Unknown Source")
                    if name:
                        response += f"- {name}\n"
            
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

# Telegram bot message handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "👋 Welcome to FactCheck Bot! Send me any message, news article, or claim, and I'll check its factual accuracy for you using advanced search technology. You can also send voice messages for fact-checking.")

@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.chat.id
    text = message.text
    
    lower_msg = text.lower()
    if lower_msg in ["join", "hello", "hi", "hey", "start"]:
        welcome_msg = "👋 Welcome to FactCheck Bot! Send me any message, news article, or claim, and I'll check its factual accuracy for you using advanced search technology. You can also send voice messages for fact-checking."
        bot.send_message(chat_id, welcome_msg)
        return
    
    # Send a "processing" message
    processing_msg = bot.send_message(chat_id, "🔍 I'm fact-checking your message. This may take a moment...")
    
    try:
        # Process the text with fact checking
        results = perform_serper_fact_check(text)
        
        formatted_response = format_serper_fact_check_results(results)
        enhanced_response = enhance_fact_check_response(formatted_response, text)
        
        # Log the response for debugging
        print(f"[TEXT] Formatted response: {enhanced_response[:100]}...")
        print(f"[TEXT] Response length: {len(enhanced_response)}")
        
        # Split into chunks if too long (Telegram has 4096 char limit)
        if len(enhanced_response) > 4000:
            chunks = [enhanced_response[i:i+4000] for i in range(0, len(enhanced_response), 4000)]
            for chunk in chunks:
                bot.send_message(chat_id, chunk, parse_mode="Markdown")
        else:
            bot.send_message(chat_id, enhanced_response, parse_mode="Markdown")
        
    except Exception as e:
        print(f"[TEXT] Error processing text: {e}")
        traceback.print_exc()
        bot.send_message(chat_id, "I encountered an error while fact-checking. Please try again with a different message.")
    
    # Delete the processing message
    bot.delete_message(chat_id, processing_msg.message_id)

@bot.message_handler(content_types=['photo', 'document', 'video', 'sticker'])
def handle_other_media(message):
    bot.reply_to(message, "I currently only fact-check text and audio messages. Please send me text or voice messages to verify.")

# Webhook routes
@app.route('/' + TELEGRAM_BOT_TOKEN, methods=['POST'])
def webhook():
    """
    Main webhook endpoint for Telegram to send updates to
    """
    print("[WEBHOOK] Received update from Telegram")
    json_str = request.get_data().decode('UTF-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return 'OK'

@app.route('/set_webhook', methods=['GET', 'POST'])
def set_webhook():
    """
    Endpoint to set the webhook URL for Telegram
    """
    webhook_url = request.args.get('url')
    if webhook_url:
        print(f"[WEBHOOK] Setting webhook to {webhook_url}")
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url + '/' + TELEGRAM_BOT_TOKEN)
        return f"Webhook set to {webhook_url}"
    return "Please provide a webhook URL"

@app.route('/remove_webhook', methods=['GET'])
def remove_webhook():
    """
    Endpoint to remove the webhook
    """
    print("[WEBHOOK] Removing webhook")
    bot.remove_webhook()
    return "Webhook removed"

@app.route('/test')
def test_route():
    """
    Test endpoint to verify Flask app is working
    """
    print("[TEST] Test route accessed")
    return "Test route works! Flask app is functioning."

@app.route('/')
def index():
    """
    Simple index page with instructions
    """
    print("[INDEX] Index page accessed")
    return """
    <h1>Telegram Fact Checking Bot</h1>
    <p>Server is running!</p>
    <h2>Setup Instructions:</h2>
    <ol>
        <li>Start ngrok with: <code>ngrok http 5002</code></li>
        <li>Get your ngrok URL (e.g., <code>https://abc123.ngrok.io</code>)</li>
        <li>Set your webhook by visiting: <code>https://abc123.ngrok.io/set_webhook?url=https://abc123.ngrok.io</code></li>
        <li>Start chatting with your bot on Telegram</li>
    </ol>
    <p>To test if the app is working, visit: <a href="/test">/test</a></p>
    """

if __name__ == '__main__':
    # Don't try to set webhook at startup - let user do it manually
    print("\n====== TELEGRAM FACT CHECKING BOT ======")
    print(f"Bot username: @{bot.get_me().username}")
    print("Server starting on port 5002")
    print("\nSETUP INSTRUCTIONS:")
    print("1. Start ngrok with: ngrok http 5002")
    print("2. Get your ngrok URL (e.g., https://abc123.ngrok.io)")
    print("3. Set webhook by visiting: your-ngrok-url/set_webhook?url=your-ngrok-url")
    print("4. Start chatting with your bot on Telegram")
    print("\nTo remove webhook, visit: your-ngrok-url/remove_webhook")
    print("To test if server is working, visit: your-ngrok-url/test")
    print("=======================================\n")
    
    # Print all available routes for debugging
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"Route: {rule}, Methods: {rule.methods}")
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)