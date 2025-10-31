import sys
import os
import base64
import io
from PIL import Image

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

def get_image_summary_from_gemini(image_path: str, api_key: str) -> str:
    def encode_image(image_path, max_size=(1024, 1024)):
        try:
            with Image.open(image_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.thumbnail(max_size)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None

    os.environ["GOOGLE_API_KEY"] = ""
    try:
        llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    except Exception as e:
        return f"Error: Could not initialize Gemini. Details: {e}"

    base64_image = encode_image(image_path)
    if not base64_image:
        return "Error: Image could not be encoded."

    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Provide a detailed one-paragraph summary of this image."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        )
        response = llm_gemini.invoke([message])
        return response.content
    except Exception as e:
        return f"Error: Gemini failed to analyze image. Details: {e}"

def get_base_post_from_pranjal(user_text_prompt: str) -> str:
    try:
        llm_ollama = ChatOllama(model="gemma:2b")
    except Exception as e:
        return f"Error: Could not initialize Ollama. Is it running? Details: {e}"

    blog_chain = LLMChain(prompt=PromptTemplate(input_variables=["description"], template="Blog: {description}"), llm=llm_ollama)
    caption_chain = LLMChain(prompt=PromptTemplate(input_variables=["description"], template="Caption: {description}"), llm=llm_ollama)
    tagline_chain = LLMChain(prompt=PromptTemplate(input_variables=["description"], template="Tagline: {description}"), llm=llm_ollama)
    
    tools = [
        Tool(name="Blog Writer", func=blog_chain.run, description="Use for blog posts."),
        Tool(name="Instagram Caption Writer", func=caption_chain.run, description="Use for social media captions."),
        Tool(name="Tagline Creator", func=tagline_chain.run, description="Use for taglines.")
    ]
    
    agent = initialize_agent(tools, llm_ollama, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)
    
    try:
        base_post = agent.run(user_text_prompt)
        return base_post
    except Exception as e:
        return f"Error: Pranjal's agent failed. Details: {e}"

def create_shubham_enhancer_chain():
    try:
        llm_ollama = ChatOllama(model="gemma:2b")
    except Exception as e:
        print(f"❌ Error initializing Ollama: {e}. Is Ollama running?")
        return None

    enhancer_prompt = PromptTemplate.from_template(
        """
        You are an expert social media post editor.
        Your job is to "enhance" a 'Base Post' by intelligently weaving in
        the details from an 'Image Description'.

        Do not just staple them together. Create a new, single, cohesive post
        that combines the *intent* of the Base Post with the *context*
        of the Image Description.
        
        ---
        Base Post (from Pranjal):
        {base_post}
        ---
        Image Description (from Saksham):
        {image_description}
        ---
        
        Final, Enhanced Post:
        """
    )
    
    shubham_chain = enhancer_prompt | llm_ollama | StrOutputParser()
    return shubham_chain

def main():
    print("====== 🚀 ORCHESTRATOR WORKFLOW STARTED 🚀 ======\n")
    
    # WARNING: Do not share this code or post it online with your key in it.
    API_KEY = ""
    
    shubham_chain = create_shubham_enhancer_chain()
    if not shubham_chain:
        sys.exit(1)
        
    print("\n--- [Orchestrator] Gathering User Inputs ---")
    image_path = input("Enter the full path to your image: ")
    user_text_prompt = input("What kind of post do you want? (e.g., 'a fun caption about my productive morning'): ")

    base_post = get_base_post_from_pranjal(user_text_prompt)
    if "Error:" in base_post:
        print(base_post)
        sys.exit(1)
    print(f"\n✅ Base Post (from Pranjal):\n{base_post}")

    image_description = get_image_summary_from_gemini(image_path, API_KEY)
    if "Error:" in image_description:
        print(image_description)
        sys.exit(1)
    print(f"\n✅ Image Description (from Saksham):\n{image_description}")

    print("\n--- [Shubham] Enhancing post with image details... ---")
    try:
        final_post = shubham_chain.invoke({
            "base_post": base_post,
            "image_description": image_description
        })
        
        print("\n====== ✨ ORCHESTRATOR WORKFLOW COMPLETE ✨ ======")
        print("Final Enhanced Post:\n")
        print(final_post)
        
    except Exception as e:
        print(f"\n--- ❌ Orchestrator Error ---")
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()
