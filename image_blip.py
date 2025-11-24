import sys
import os
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ==========================================================
# 1. STABLE LANGCHAIN IMPORTS (FINAL CORRECTED VERSION) 🐍
# ==========================================================
# Core abstractions from langchain-core (e.g., PromptTemplate, Tool, OutputParser)
from langchain_core.prompts import PromptTemplate 
from langchain_core.tools import tool, Tool 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough

# AGENT COMPONENTS: Imported from their exact locations to avoid path errors
# AgentExecutor (The runtime loop) is in Core
from langchain.agents import AgentExecutor

# create_react_agent (The factory function) is in the specific agent module
# This is the deep path required for your specific 1.0.3 installation
from langchain.agents import create_react_agent 

# Integration components from langchain-community
from langchain_community.chat_models import ChatOllama

# --- BLIP Imports ---
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================================
# 2. LOAD LOCAL MODELS (BLIP + OLLAMA)
# ==========================================================

# --- Load BLIP model globally ---
try:
    print("⏳ Loading BLIP model (Salesforce/blip-image-captioning-base)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("✅ BLIP model loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error: Could not load BLIP model. {e}")
    print("   Please ensure 'torch' and 'transformers' are installed.")
    blip_processor = None
    blip_model = None

# --- Load Ollama LLM globally ---
try:
    print("⏳ Connecting to Ollama (model=mistral)...")
    ollama_llm = ChatOllama(model="mistral")
    # Quick test connection
    ollama_llm.invoke("test connection") 
    print("✅ Ollama connection successful.")
except Exception as e:
    print(f"❌ Critical Error: Could not connect to Ollama. Is Ollama running? {e}")
    ollama_llm = None

# ==========================================================
# 3. FLASK APP SETUP
# ==========================================================

# Correct: Use __name__
app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/')
def serve_html():
    # Serves the 'app.html' file from the same directory
    return send_from_directory('.', 'app.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    # Serves other files like CSS or JS
    return send_from_directory('.', filename)

# ==========================================================
# 4. AGENT & CHAIN DEFINITIONS
# ==========================================================

# --- AGENT 2: BLIP TOOL DEFINITION (STABLE @tool decorator) ---
@tool
def blip_image_analyzer_tool(image_path: str) -> str:
    """
    Analyzes an image to extract a raw caption using the local BLIP model. 
    Input is the local file path to the image. 
    The output is a single-sentence caption.
    """
    if not blip_processor or not blip_model:
        return "Error: BLIP model is not loaded."
        
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return "Error: Image file not found."
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return "Error: Could not open image."

    try:
        inputs = blip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            # Ensure model is on CPU if not using a CUDA device
            if torch.cuda.is_available():
                 inputs = blip_processor(images=image, return_tensors="pt").to("cuda")
                 blip_model.to("cuda")
            
            out = blip_model.generate(**inputs, max_new_tokens=75) 
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"❌ Error during BLIP generation: {e}")
        return f"Error: BLIP failed to analyze image. Details: {e}"

# --- AGENT 2: BLIP ORCHESTRATOR (PROFESSIONAL AGENT) ---
def get_image_description_via_agent(image_path: str) -> str:
    """
    Initializes and runs the LangChain Agent for image analysis.
    Uses the modern LCEL agent construction method via create_react_agent.
    """
    if not ollama_llm:
        return "Error: Ollama is not available."

    tools = [blip_image_analyzer_tool]
    tool_names = [tool.name for tool in tools]

    # --- PROFESSIONAL SYSTEM PROMPT (ReAct Format) ---
    prompt_template_content = """
        You are a professional Visual Semantic Analyst. Your task is to process a raw image description 
        (obtained from the 'blip_image_analyzer_tool') and translate it into **semantic insights** focused on creative content generation.

        Your analysis must include:
        1.  **Key Content & Objects:** The central subjects and detailed composition.
        2.  **Emotional Context & Tone:** The implied feeling, mood, or atmosphere (e.g., peaceful, energetic, suspenseful).
        3.  **Creative Vibe/Style:** Suggestive terms for the final post's tone (e.g., minimalist, vibrant, nostalgic).

        You must first use the tool on the provided file path and then generate the analysis.
        The final output should be a single, detailed paragraph summarizing the three points above, and nothing else.

        TOOLS:
        ------
        You have access to the following tools:
        {tool_names}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(
        prompt_template_content
    ).partial(tool_names=", ".join(tool_names))

    try:
        # 1. Create the Agent (Prompt + LLM + Parser)
        image_agent = create_react_agent(ollama_llm, tools, prompt) 

        # 2. Create the Agent Executor Chain (LCEL Runnable)
        agent_executor = AgentExecutor(agent=image_agent, tools=tools, verbose=False)

        # The prompt is the task for the agent
        prompt_input = f"Analyze the image located at this file path: {image_path}."
        
        # 3. Run the agent and get the final professional description
        result = agent_executor.invoke({"input": prompt_input})
        
        return result['output']
        
    except Exception as e:
        print(f"❌ Error in BLIP Agent execution: {e}")
        return f"Error: BLIP Agent failed to run. Details: {e}"


# --- AGENT 1: PRANJAL (TEXT) - LCEL Chains for Tools ---
def get_base_post_from_pranjal(user_text_prompt: str) -> str:
    """
    Agent "Pranjal": Generates a base post (blog, caption, or tagline)
    based on the user's text prompt. Uses modern LCEL agent construction.
    """
    if not ollama_llm:
        return "Error: Ollama is not available."

    # Using LCEL (Runnable) chains as stable alternatives to LLMChain
    def create_lcel_chain(template):
        # This creates a simple prompt|model|parser chain
        return PromptTemplate.from_template(template) | ollama_llm | StrOutputParser()

    blog_chain = create_lcel_chain("Write a creative blog post about: {description}")
    caption_chain = create_lcel_chain("Write an engaging Instagram caption for: {description}")
    tagline_chain = create_lcel_chain("Create a catchy tagline for: {description}")
    
    tools = [
        # Tool is imported from langchain_core
        Tool(name="Blog_Writer", func=lambda desc: blog_chain.invoke({"description": desc}), description="Use this to write a blog post. Input should be the topic."),
        Tool(name="Instagram_Caption_Writer", func=lambda desc: caption_chain.invoke({"description": desc}), description="Use this to write a social media caption. Input should be a description of the post."),
        Tool(name="Tagline_Creator", func=lambda desc: tagline_chain.invoke({"description": desc}), description="Use this to create a short tagline. Input should be the product or idea.")
    ]
    tool_names = [tool.name for tool in tools]
    
    # --- Standard ReAct Prompt for Agent Pranjal ---
    pranjal_prompt_template = PromptTemplate.from_template(
        """
        You are a creative content generator. Your goal is to write a piece of creative text 
        (blog post, caption, or tagline) based on the user's request using the appropriate tool.
        You have access to the following tools: {tool_names}.

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
    ).partial(tool_names=", ".join(tool_names))

    try:
        # 1. Create the Agent
        agent = create_react_agent(ollama_llm, tools, pranjal_prompt_template)
        
        # 2. Create the Agent Executor Chain
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        # 3. Run the agent
        result = agent_executor.invoke({"input": user_text_prompt})
        return result['output']
    except Exception as e:
        print(f"❌ Error in Pranjal agent: {e}")
        return f"Error: Pranjal's agent failed. Details: {e}"

# --- AGENT 3: SHUBHAM (ENHANCER CHAIN) ---
def create_shubham_enhancer_chain():
    """
    Chain "Shubham": Takes the base post and image description
    and merges them into one cohesive post.
    """
    if not ollama_llm:
        return None

    enhancer_prompt = PromptTemplate.from_template(
        """
        You are an expert social media post editor.
        Your job is to "enhance" a 'Base Post' by intelligently weaving in
        the details from an 'Image Description'.

        The Image Description contains detailed semantic insights (content, emotion, style).
        Use these insights to refine the tone, vocabulary, and specific details of the Base Post.

        ---
        Base Post (from Pranjal):
        {base_post}
        ---
        Image Description (from BLIP Agent):
        {image_description}
        ---
        
        Final, Enhanced Post:
        """
    )
    
    # Define the chain using LangChain Expression Language (LCEL)
    shubham_chain = enhancer_prompt | ollama_llm | StrOutputParser()
    return shubham_chain

# ==========================================================
# 5. API ENDPOINT (ORCHESTRATOR)
# ==========================================================

@app.route("/api/orchestrate", methods=["POST"])
def orchestrate():
    print("🔥 Request received at /api/orchestrate")
    
    if not ollama_llm or not blip_model:
        return jsonify({"error": "A core model (Ollama or BLIP) failed to load. Check server logs."}), 500
        
    temp_image_path = None
    try:
        user_text_prompt = request.form.get("prompt")
        image_file = request.files.get("image")
        if not image_file or not user_text_prompt:
            return jsonify({"error": "Missing image or prompt"}), 400

        # Save image temporarily
        temp_image_path = "temp_upload.jpg"
        image_file.save(temp_image_path)

        # --- Initialize Enhancer Chain ---
        shubham_chain = create_shubham_enhancer_chain()
        if not shubham_chain:
            return jsonify({"error": "Failed to initialize enhancer chain"}), 500

        # --- Agent 1: Pranjal (Text Generation) ---
        print("🚀 Starting Agent 1 (Pranjal) for text prompt...")
        base_post = get_base_post_from_pranjal(user_text_prompt)
        if "Error:" in base_post:
            return jsonify({"error": base_post}), 500
        print(f"✅ Agent 1 finished.")

        # --- Agent 2: BLIP (Image Analysis via dedicated Agent) ---
        print("🚀 Starting Agent 2 (BLIP) for professional image analysis...")
        image_description = get_image_description_via_agent(temp_image_path)
        if "Error:" in image_description:
            return jsonify({"error": image_description}), 500
        print(f"✅ Agent 2 finished. Analysis generated.")

        # --- Agent 3: Shubham (Enhancer Chain) ---
        print("🚀 Starting Agent 3 (Shubham) to enhance post...")
        final_post = shubham_chain.invoke({
            "base_post": base_post,
            "image_description": image_description
        })
        print(f"✅ Agent 3 finished.")

        # Return the full result
        return jsonify({
            "base_post": base_post,
            "image_description": image_description,
            "final_post": final_post
        })

    except Exception as e:
        print(f"❌ Unhandled error in /api/orchestrate: {e}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up the temporary image file in all cases
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ==========================================================
# 6. RUN APP
# ==========================================================

# Correct: Use __name__ and port 5001 (to avoid conflict)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
