"""
Flask API for Multi-Agent Article Debate System

This API allows users to interact with a multi-agent debate system through HTTP requests.
The system analyzes articles using multiple LLM agents and facilitates a debate between them.
"""

import os
from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict, NotRequired
import enum
import json
import requests
import traceback
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Configure API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# Define intent types for user input
class UserIntent(enum.Enum):
    GENERAL_COMMENT = "general_comment"
    DISAGREE = "disagree"
    REQUEST_EVIDENCE = "request_evidence"
    CLARIFY = "clarify"
    ADD_INFORMATION = "add_information"
    ASK_QUESTION = "ask_question"
    CHALLENGE_FACT = "challenge_fact"
    END_DEBATE = "end_debate"

# Define state class
class DebateState(TypedDict):
    """State for the debate including article and agents' analysis"""
    debate_id: str  # Unique identifier for the debate
    article: str  # The input article
    messages: List[Dict]  # Conversation history
    summary: NotRequired[Optional[str]]  # Reader's summary
    pro_arguments: NotRequired[List[str]]  # Arguments in favor
    con_arguments: NotRequired[List[str]]  # Arguments against
    facts_checked: NotRequired[Dict[str, bool]]  # Facts and their verification status
    current_agent: NotRequired[str]  # Current active agent
    next_agent: NotRequired[str]  # Next agent to run
    debate_complete: NotRequired[bool]  # Whether the debate is complete
    user_input: NotRequired[Optional[str]]  # Latest user input
    user_intent: NotRequired[Optional[str]]  # Detected intent of user input
    last_mentioned_topic: NotRequired[Optional[str]]  # Track what was last discussed

# Store active debates in memory
# In a production environment, use a database
active_debates = {}

# Define tools
@tool
def google_fact_check(claim: str) -> Dict[str, Any]:
    """
    Check a claim against the Google Fact Check API
    
    Args:
        claim: The statement to verify
        
    Returns:
        Dict: Fact check results or error message
    """
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "key": GOOGLE_FACT_CHECK_API_KEY,
            "query": claim
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "verified": False}

# Define agent prompts
SUPERVISOR_PROMPT = """
You are the supervisor of a multi-agent debate about an article. 
Your role is to coordinate between the reader, pro_writer, con_writer, and fact_checker agents.
You'll decide which agent should go next based on the current state of the debate.

Reader: Summarizes and extracts key points from the article
Pro Writer: Makes arguments supporting the article's viewpoints
Con Writer: Makes arguments against the article's viewpoints
Fact Checker: Verifies factual claims made by either side
User: The human participant who can add their own perspective

Follow these rules:
1. Start with the reader to analyze the article
2. Alternate between pro_writer and con_writer, allowing for informed debate
3. Consult the fact_checker when factual claims need verification
4. When the user contributes, acknowledge their input and route to the appropriate agent:
   - If they disagree with a point, have con_writer respond
   - If they request evidence, have fact_checker respond
   - If they ask for clarification, have the appropriate agent respond
   - If they add new information, acknowledge it and decide who should respond
5. Make sure the debate stays balanced and each side gets equal speaking time
6. Keep your responses short and focused on directing the debate to the next appropriate agent
"""

READER_PROMPT = """
You are a thorough article reader and analyzer. 
Your role is to carefully read the provided article and extract:
1. The main thesis or argument
2. Key supporting points
3. Evidence presented
4. Any notable rhetoric or persuasion techniques used
5. The target audience

Be balanced and objective in your analysis. Do not take a side or inject your own opinions.
Present the article's viewpoints as they are, whether you agree with them or not.
Provide your analysis in a structured format that will be useful for debaters on both sides.

If the user asks for clarification, explain your analysis in simpler terms.
If the user provides additional context, incorporate it into your analysis.
"""

PRO_WRITER_PROMPT = """
You are a debater arguing IN FAVOR of the viewpoints presented in the article.
Your goal is to strengthen and defend the article's key arguments and perspectives.

Use these strategies:
1. Elaborate on the strongest points from the article
2. Add additional supporting evidence and examples
3. Address potential weaknesses and counter them
4. Use persuasive language while maintaining intellectual honesty
5. Consider how these viewpoints could benefit society

If the user challenges your arguments, acknowledge their perspective but maintain your position.
If the user requests evidence, provide specific data or refer to the fact checker.
If the user asks for clarification, explain your points in a more accessible way.

Be persuasive but not misleading. Stick to logical arguments and evidence.
"""

CON_WRITER_PROMPT = """
You are a debater arguing AGAINST the viewpoints presented in the article.
Your goal is to critique and challenge the article's key arguments and perspectives.

Use these strategies:
1. Identify logical fallacies or weak reasoning
2. Present counter-evidence and alternative perspectives
3. Challenge assumptions made in the article
4. Offer alternative frameworks for viewing the issue
5. Consider potential negative consequences of the article's viewpoints

If the user supports your arguments, acknowledge their input while remaining objective.
If the user challenges your points, respond with respect while maintaining your position.
If the user asks for clarification, explain your critique more clearly.

Be persuasive but not dismissive. Stick to logical arguments and evidence.
"""

FACT_CHECKER_PROMPT = """
You are a fact checker verifying claims made in a debate about an article.
Your goal is to determine the factual accuracy of statements using credible sources.

For each claim you check:
1. Use the Google Fact Check API to find existing fact checks
2. Evaluate the reliability of sources
3. Report the factual status: Supported, Unsupported, Disputed, or Needs More Context
4. Provide the evidence justifying your conclusion

If the user challenges your fact check, explain your methodology and sources.
If the user provides additional evidence, evaluate it and update your assessment if needed.

Be thorough, accurate, and impartial. Your aim is truth, not supporting either side.
"""

# Intent detection function
def detect_user_intent(user_input: str, current_context: Dict) -> UserIntent:
    """Analyze user input to determine their intent"""
    user_input = user_input.lower()
    
    # Check for debate termination
    if any(word in user_input for word in ["done", "exit", "quit", "stop", "end"]):
        return UserIntent.END_DEBATE
    
    # Check for disagreement
    if any(word in user_input for word in ["disagree", "incorrect", "wrong", "not true", "false", "mistaken"]):
        return UserIntent.DISAGREE
    
    # Check for evidence requests
    if any(phrase in user_input for phrase in ["evidence", "proof", "source", "citation", "verify", "fact check", "support this"]):
        return UserIntent.REQUEST_EVIDENCE
    
    # Check for clarification requests
    if any(phrase in user_input for phrase in ["clarify", "explain", "what do you mean", "don't understand", "confused", "unclear"]):
        return UserIntent.CLARIFY
    
    # Check for information addition
    if any(phrase in user_input for phrase in ["also", "addition", "add", "moreover", "furthermore", "consider", "point out"]):
        return UserIntent.ADD_INFORMATION
    
    # Check for questions
    if "?" in user_input or any(word in user_input for word in ["who", "what", "where", "when", "why", "how"]):
        return UserIntent.ASK_QUESTION
        
    # Check for fact challenges
    if any(phrase in user_input for phrase in ["actually", "in fact", "the truth is", "that's not accurate", "that's incorrect"]):
        return UserIntent.CHALLENGE_FACT
    
    # Default to general comment
    return UserIntent.GENERAL_COMMENT

# Helper functions for agents
def generate_agent_response(agent_type, state):
    """Generate a response for an agent based on the current state"""
    
    # Select the appropriate prompt for the agent
    if agent_type == "supervisor":
        prompt = SUPERVISOR_PROMPT
    elif agent_type == "reader":
        prompt = READER_PROMPT
    elif agent_type == "pro_writer":
        prompt = PRO_WRITER_PROMPT
    elif agent_type == "con_writer":
        prompt = CON_WRITER_PROMPT
    elif agent_type == "fact_checker":
        prompt = FACT_CHECKER_PROMPT
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Prepare input messages
    messages = [SystemMessage(content=prompt)]
    
    # Add context based on the agent type
    if agent_type == "reader":
        article = state.get("article", "")
        messages.append(HumanMessage(content=f"Please analyze this article: {article}"))
    elif agent_type == "pro_writer":
        summary = state.get("summary", "")
        user_intent = state.get("user_intent")
        user_input = state.get("user_input", "")
        
        content = f"Here's the article summary: {summary}\n\nPlease provide arguments IN FAVOR of the viewpoints in the article."
        
        if user_intent and user_input:
            content += f"\n\nThe user has said: '{user_input}'"
            if user_intent == UserIntent.DISAGREE.value:
                content += "\nPlease address their disagreement while maintaining your position."
            elif user_intent == UserIntent.REQUEST_EVIDENCE.value:
                content += "\nPlease provide evidence for your claims."
            elif user_intent == UserIntent.CLARIFY.value:
                content += "\nPlease clarify your position in simpler terms."
                
        messages.append(HumanMessage(content=content))
        
    elif agent_type == "con_writer":
        summary = state.get("summary", "")
        pro_args = state.get("pro_arguments", [])
        last_pro = pro_args[-1] if pro_args else ""
        user_intent = state.get("user_intent")
        user_input = state.get("user_input", "")
        
        content = f"Here's the article summary: {summary}\n\nThe pro writer argued: {last_pro}\n\nPlease provide arguments AGAINST the viewpoints in the article."
        
        if user_intent and user_input:
            content += f"\n\nThe user has said: '{user_input}'"
            if user_intent == UserIntent.DISAGREE.value:
                content += "\nPlease address their disagreement while maintaining your position."
            elif user_intent == UserIntent.REQUEST_EVIDENCE.value:
                content += "\nPlease provide evidence for your claims."
            elif user_intent == UserIntent.CLARIFY.value:
                content += "\nPlease clarify your position in simpler terms."
                
        messages.append(HumanMessage(content=content))
        
    elif agent_type == "fact_checker":
        pro_args = state.get("pro_arguments", [])
        con_args = state.get("con_arguments", [])
        last_pro = pro_args[-1] if pro_args else ""
        last_con = con_args[-1] if con_args else ""
        user_intent = state.get("user_intent")
        user_input = state.get("user_input", "")
        
        content = f"Please fact check these arguments:\n\nPro: {last_pro}\n\nCon: {last_con}"
        
        if user_intent and user_input:
            content += f"\n\nThe user has said: '{user_input}'"
            if user_intent == UserIntent.CHALLENGE_FACT.value:
                content += "\nPlease address their challenge to your fact check."
            elif user_intent == UserIntent.REQUEST_EVIDENCE.value:
                content += "\nPlease provide additional evidence for your fact check."
                
        messages.append(HumanMessage(content=content))
        
    elif agent_type == "supervisor":
        # For supervisor, include context about the user's intent
        current_agent = state.get("current_agent", None)
        user_intent = state.get("user_intent")
        user_input = state.get("user_input", "")
        
        if user_intent and user_input:
            if user_intent == UserIntent.END_DEBATE.value:
                messages.append(HumanMessage(content="The user wants to end the debate. Please conclude the discussion."))
            else:
                intent_desc = user_intent.replace("_", " ") if isinstance(user_intent, str) else "comment"
                messages.append(HumanMessage(content=f"The user has provided a {intent_desc}: '{user_input}'\nWhich agent should respond to this?"))
        elif current_agent:
            messages.append(HumanMessage(content=f"The {current_agent} just finished. Which agent should go next?"))
        else:
            messages.append(HumanMessage(content="Let's start the debate. Which agent should go first?"))
    
    # Add relevant conversation history for context
    if state.get("messages"):
        # Add only the last few messages to avoid context window issues
        relevant_messages = state["messages"][-5:]
        for msg in relevant_messages:
            if isinstance(msg, dict):
                # Convert dict to Message object
                if msg.get("type") == "human":
                    messages.append(HumanMessage(content=msg.get("content", ""), name=msg.get("name", "user")))
                else:
                    messages.append(AIMessage(content=msg.get("content", ""), name=msg.get("name", "ai")))
            elif not isinstance(msg, SystemMessage):  # Avoid duplicate system messages
                messages.append(msg)
    
    # Generate response with timeout and retries
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response for {agent_type} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:  # Last attempt failed
                return f"[Error generating {agent_type} response. Continuing debate...]"
            # Wait briefly before retrying
            import time
            time.sleep(1)

# Agent execution functions
def run_supervisor(state):
    """Run the supervisor agent and determine next steps"""
    # Initialize required state fields if not present
    if "summary" not in state:
        state["summary"] = None
    if "pro_arguments" not in state:
        state["pro_arguments"] = []
    if "con_arguments" not in state:
        state["con_arguments"] = []
    if "facts_checked" not in state:
        state["facts_checked"] = {}
    if "debate_complete" not in state:
        state["debate_complete"] = False
    
    # Check if user wants to end the debate
    if state.get("user_intent") == UserIntent.END_DEBATE.value:
        state["debate_complete"] = True
        state["messages"].append({
            "type": "ai",
            "name": "supervisor",
            "content": "The debate has ended. Thank you for participating!"
        })
        return state
    
    # If we haven't analyzed the article yet, start with the reader
    if not state.get("summary"):
        supervisor_message = "Let's begin by having our reader analyze the article in detail."
        state["messages"].append({
            "type": "ai",
            "name": "supervisor",
            "content": supervisor_message
        })
        state["current_agent"] = "supervisor"
        state["next_agent"] = "reader"
        return state
    
    # Generate supervisor message
    supervisor_message = generate_agent_response("supervisor", state)
    state["messages"].append({
        "type": "ai",
        "name": "supervisor",
        "content": supervisor_message
    })
    
    # Determine next agent based on supervisor's message and user intent
    user_intent = state.get("user_intent")
    current_agent = state.get("current_agent", "")
    next_agent = "reader"  # Default
    
    # Route based on user intent if present
    if user_intent:
        if user_intent == UserIntent.REQUEST_EVIDENCE.value:
            next_agent = "fact_checker"
        elif user_intent == UserIntent.DISAGREE.value:
            # If they disagree with pro, go to con; if they disagree with con, go to pro
            if current_agent == "pro_writer":
                next_agent = "con_writer"
            elif current_agent == "con_writer":
                next_agent = "pro_writer"
            else:
                next_agent = "con_writer"  # Default if unclear
        elif user_intent == UserIntent.CLARIFY.value:
            # Keep the same agent to clarify
            next_agent = current_agent if current_agent not in ["supervisor", None] else "reader"
        elif user_intent == UserIntent.CHALLENGE_FACT.value:
            next_agent = "fact_checker"
        else:
            # For other intents, look for clues in the supervisor's message
            if "reader" in supervisor_message.lower() or "article" in supervisor_message.lower():
                next_agent = "reader"
            elif "pro writer" in supervisor_message.lower() or "in favor" in supervisor_message.lower():
                next_agent = "pro_writer"
            elif "con writer" in supervisor_message.lower() or "against" in supervisor_message.lower():
                next_agent = "con_writer"
            elif "fact check" in supervisor_message.lower() or "verify" in supervisor_message.lower():
                next_agent = "fact_checker"
    else:
        # No user intent, use standard flow or hints from supervisor
        if "reader" in supervisor_message.lower() or "article" in supervisor_message.lower():
            next_agent = "reader"
        elif "pro writer" in supervisor_message.lower() or "in favor" in supervisor_message.lower():
            next_agent = "pro_writer"
        elif "con writer" in supervisor_message.lower() or "against" in supervisor_message.lower():
            next_agent = "con_writer"
        elif "fact check" in supervisor_message.lower() or "verify" in supervisor_message.lower():
            next_agent = "fact_checker"
        elif current_agent == "reader":
            next_agent = "pro_writer"
        elif current_agent == "pro_writer":
            next_agent = "con_writer"
        elif current_agent == "con_writer":
            next_agent = "fact_checker"
        elif current_agent == "fact_checker":
            next_agent = "pro_writer"
    
    # Clear user input and intent after processing
    state["user_input"] = None
    state["user_intent"] = None
    state["current_agent"] = "supervisor"
    state["next_agent"] = next_agent
    
    return state

def run_reader(state):
    """Run the reader agent to analyze the article"""
    article = state.get("article", "")
    if not article:
        response_content = "No article provided for analysis."
    else:
        response_content = generate_agent_response("reader", state)
    
    # Store summary in state
    state["summary"] = response_content
    
    # Add reader's message to the conversation
    state["messages"].append({
        "type": "ai",
        "name": "reader", 
        "content": response_content
    })
    
    # Update current agent
    state["current_agent"] = "reader"
    state["next_agent"] = "supervisor"
    state["last_mentioned_topic"] = "article analysis"
    
    return state

def run_pro_writer(state):
    """Run the pro writer agent to argue in favor of the article"""
    summary = state.get("summary", "")
    if not summary:
        response_content = "I need an article summary to form arguments."
    else:
        response_content = generate_agent_response("pro_writer", state)
    
    # Store pro arguments in state
    state["pro_arguments"].append(response_content)
    
    # Add pro writer's message to the conversation
    state["messages"].append({
        "type": "ai",
        "name": "pro_writer",
        "content": response_content
    })
    
    # Update current agent
    state["current_agent"] = "pro_writer"
    state["next_agent"] = "supervisor"
    state["last_mentioned_topic"] = "arguments in favor"
    
    return state

def run_con_writer(state):
    """Run the con writer agent to argue against the article"""
    summary = state.get("summary", "")
    if not summary:
        response_content = "I need an article summary to form counter-arguments."
    else:
        response_content = generate_agent_response("con_writer", state)
    
    # Store con arguments in state
    state["con_arguments"].append(response_content)
    
    # Add con writer's message to the conversation
    state["messages"].append({
        "type": "ai", 
        "name": "con_writer", 
        "content": response_content
    })
    
    # Update current agent
    state["current_agent"] = "con_writer"
    state["next_agent"] = "supervisor"
    state["last_mentioned_topic"] = "arguments against"
    
    return state

def run_fact_checker(state):
    """Run the fact checker agent to verify claims"""
    # Get the latest arguments to check
    latest_pro = state["pro_arguments"][-1] if state["pro_arguments"] else ""
    latest_con = state["con_arguments"][-1] if state["con_arguments"] else ""
    
    if not latest_pro and not latest_con:
        response_content = "No arguments to fact check yet."
    else:
        response_content = generate_agent_response("fact_checker", state)
    
    # Add fact checker's message to the conversation
    state["messages"].append({
        "type": "ai",
        "name": "fact_checker",
        "content": response_content
    })
    
    # Simple parsing of fact check results
    facts_checked = state.get("facts_checked", {})
    fact_count = len(facts_checked) + 1
    fact_key = f"fact_check_{fact_count}"
    
    if "supported" in response_content.lower():
        facts_checked[fact_key] = True
    elif "unsupported" in response_content.lower():
        facts_checked[fact_key] = False
    else:
        facts_checked[fact_key] = "inconclusive"
    
    state["facts_checked"] = facts_checked
    
    # Update current agent
    state["current_agent"] = "fact_checker"
    state["next_agent"] = "supervisor"
    state["last_mentioned_topic"] = "fact checking"
    
    return state

def process_debate_step(debate_id):
    """Process one step of the debate"""
    if debate_id not in active_debates:
        return {"error": "Debate not found"}, 404
    
    state = active_debates[debate_id]
    
    if state.get("debate_complete", False):
        return {
            "debate_id": debate_id,
            "message": "Debate is already complete",
            "debate_complete": True,
            "current_agent": state.get("current_agent"),
            "latest_message": state["messages"][-1] if state["messages"] else None
        }
    
    # Determine which agent to run based on next_agent
    agent_to_run = state.get("next_agent", "supervisor")
    
    # Execute the appropriate agent
    try:
        if agent_to_run == "supervisor":
            state = run_supervisor(state)
        elif agent_to_run == "reader":
            state = run_reader(state)
        elif agent_to_run == "pro_writer":
            state = run_pro_writer(state)
        elif agent_to_run == "con_writer":
            state = run_con_writer(state)
        elif agent_to_run == "fact_checker":
            state = run_fact_checker(state)
        else:
            # Default to supervisor if unknown agent
            state = run_supervisor(state)
    except Exception as e:
        print(f"Error during debate step: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error processing debate step: {str(e)}"}, 500
    
   
    active_debates[debate_id] = state
    
    latest_message = state["messages"][-1] if state["messages"] else None
    
    return {
        "debate_id": debate_id,
        "current_agent": state.get("current_agent"),
        "next_agent": state.get("next_agent"),
        "latest_message": latest_message,
        "debate_complete": state.get("debate_complete", False),
        "waiting_for_user": state.get("current_agent") != "supervisor"
    }

# API Routes
@app.route('/api/debates', methods=['POST'])
def create_debate():
    """Create a new debate with an article"""
    data = request.json
    
    if not data or 'article' not in data:
        return jsonify({"error": "Missing 'article' field in request"}), 400
    
    article = data['article']
    
    debate_id = str(uuid.uuid4())
    
    state = {
        "debate_id": debate_id,
        "article": article,
        "messages": [{
            "type": "human",
            "name": "system",
            "content": "Let's begin a debate about the article."
        }],
        "summary": None,
        "pro_arguments": [],
        "con_arguments": [],
        "facts_checked": {},
        "current_agent": None,
        "next_agent": "supervisor",
        "debate_complete": False,
        "user_input": None,
        "user_intent": None,
        "last_mentioned_topic": None
    }
    
    active_debates[debate_id] = state
    
    result = process_debate_step(debate_id)
    
    return jsonify({
        "debate_id": debate_id, 
        "message": "Debate created successfully",
        "current_agent": result.get("current_agent"),
        "next_agent": result.get("next_agent"),
        "latest_message": result.get("latest_message"),
        "waiting_for_user": result.get("waiting_for_user", False)
    })

@app.route('/api/debates/<debate_id>/next', methods=['GET'])
def next_debate_step(debate_id):
    """Process the next step in the debate"""
    result = process_debate_step(debate_id)
    
    if isinstance(result, tuple):  
        return jsonify(result[0]), result[1]
    
    return jsonify(result)

@app.route('/api/debates/<debate_id>/input', methods=['POST'])
def submit_user_input(debate_id):
    """Submit user input to an ongoing debate"""
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404
    
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "Missing 'input' field in request"}), 400
    
    state = active_debates[debate_id]
    
    if state.get("debate_complete", False):
        return jsonify({
            "debate_id": debate_id,
            "message": "Debate is already complete",
            "debate_complete": True
        })
    
    user_input = data['input']
    

    if user_input.lower() in ["done", "exit", "quit", "stop", "end"]:
        state["user_intent"] = UserIntent.END_DEBATE.value
        state["user_input"] = user_input
        state["next_agent"] = "supervisor" 
    else:
        intent = detect_user_intent(user_input, state)
        state["user_intent"] = intent.value
        
  
        state["messages"].append({
            "type": "human",
            "name": "user",
            "content": user_input
        })
        state["user_input"] = user_input
        state["next_agent"] = "supervisor" 
    
    active_debates[debate_id] = state
    
    result = process_debate_step(debate_id)
    
    return jsonify(result)

@app.route('/api/debates/<debate_id>', methods=['GET'])
def get_debate_status(debate_id):
    """Get the current status of a debate"""
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404
    
    state = active_debates[debate_id]
    
    return jsonify({
        "debate_id": debate_id,
        "current_agent": state.get("current_agent"),
        "next_agent": state.get("next_agent"),
        "debate_complete": state.get("debate_complete", False),
        "message_count": len(state.get("messages", [])),
        "summary_available": state.get("summary") is not None,
        "pro_arguments_count": len(state.get("pro_arguments", [])),
        "con_arguments_count": len(state.get("con_arguments", [])),
        "facts_checked_count": len(state.get("facts_checked", {})),
        "waiting_for_user": state.get("current_agent") != "supervisor"
    })

@app.route('/api/debates/<debate_id>/messages', methods=['GET'])
def get_debate_messages(debate_id):
    """Get all messages from a debate"""
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404
    
    state = active_debates[debate_id]
    
    return jsonify({
        "debate_id": debate_id,
        "messages": state.get("messages", []),
        "total_messages": len(state.get("messages", [])),
        "current_agent": state.get("current_agent"),
        "next_agent": state.get("next_agent"),
        "debate_complete": state.get("debate_complete", False),
        "waiting_for_user": state.get("current_agent") != "supervisor"
    })

@app.route('/api/debates/<debate_id>/summary', methods=['GET'])
def get_debate_summary(debate_id):
    """Get a summary of the debate including stats and outcomes"""
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404
    
    state = active_debates[debate_id]
    
    return jsonify({
        "debate_id": debate_id,
        "article_length": len(state.get("article", "")),
        "total_messages": len(state.get("messages", [])),
        "pro_arguments_count": len(state.get("pro_arguments", [])),
        "con_arguments_count": len(state.get("con_arguments", [])),
        "facts_checked_count": len(state.get("facts_checked", {})),
        "user_contributions": sum(1 for m in state.get("messages", []) if m.get("type") == "human" and m.get("name") == "user"),
        "debate_complete": state.get("debate_complete", False),
        "debate_duration": f"{len(state.get('messages', []))} turns"
    })

@app.route('/api/debates/<debate_id>', methods=['DELETE'])
def delete_debate(debate_id):
    """Delete a debate session"""
    if debate_id not in active_debates:
        return jsonify({"error": "Debate not found"}), 404
    
    # Remove the debate from active_debates
    del active_debates[debate_id]
    
    return jsonify({
        "debate_id": debate_id,
        "message": "Debate successfully deleted"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "active_debates": len(active_debates),
        "version": "1.0.0"
    })

# Clean up debates periodically (optional - for a production app)
def cleanup_inactive_debates():
    """Periodically remove inactive debates to free up memory"""
    # This could be run by a scheduler in a production app
    # For now, we'll manually clean up if there are too many debates
    if len(active_debates) > 100:  # Arbitrary limit
        # Remove oldest debates
        sorted_debates = sorted(active_debates.items(), key=lambda x: len(x[1]["messages"]))
        # Keep only the 50 most active debates
        debates_to_remove = sorted_debates[:50]
        for debate_id, _ in debates_to_remove:
            del active_debates[debate_id]

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)