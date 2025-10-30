import sys
import os
import base64
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image

# --- 1. Setup the API Key ---
# ⚠️ PASTE YOUR KEY HERE
os.environ["GOOGLE_API_KEY"] = "---"

if "YOUR_GEMINI_API_KEY" in os.environ.get("GOOGLE_API_KEY", ""):
    print("❌ Error: Please replace 'YOUR_GEMINI_API_KEY_GOES_HERE' with your actual API key.")
    print("You can get a free key from https://aistudio.google.com/")
    sys.exit(1)

# --- 2. Function to encode the image ---
def encode_image(image_path, max_size=(1024, 1024)):
    """
    Loads an image, converts to RGB, resizes, and encodes as base64.
    """
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

# --- 3. Setup the Multimodal LLM ---
try:
    # Note: You've chosen 'gemini-2.5-flash'. 
    # 'gemini-1.5-flash' is also a great, reliable option for the free tier.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
    llm.invoke("Test") # Quick test
    print("✅ Successfully connected to Google Gemini API.")
except Exception as e:
    print(f"--- ❌ Error initializing Google Gemini ---")
    print(f"Error: {e}")
    print("\nPlease check your API key and 'pip install langchain-google-genai'.")
    sys.exit(1)

# --- 4. Run the Captioning ---
# ⚠️ This is the path to your image
IMAGE_PATH = "C:/Users/SHUBHAM SAHU/OneDrive - UPES/Internships/ARP/caption_gen/sunrise.jpg"

print(f"⏳ Loading and encoding image: {IMAGE_PATH}")
base64_image = encode_image(IMAGE_PATH)

if base64_image:
    try:
        # Create the prompt. We pass a list of content parts.
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Provide a detailed one-paragraph summary of this image." 
                },
                {
                    "type": "image_url",
                    # This is the correct "data URI" format for Gemini
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        )

        # Invoke the model
        print("\n⏳ Contacting Gemini for a caption...")
        response = llm.invoke([message])
        
        print("\n✨ --- Generated Caption --- ✨\n")
        print(response.content)
#pip install --upgrade langchain-google-genai pillow
    except Exception as e:
        print(f"\n❌ Error during model invocation: {e}")
else:
    print("Exiting because image could not be loaded or encoded.")
