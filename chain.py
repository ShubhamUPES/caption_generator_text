from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

system_prompts = {
    "blog": (
        "You are a professional blog writer. "
        "Write an engaging short blog paragraph about the given description. "
        "Be descriptive and informative."
    ),
    "caption": (
        "You are an Instagram caption writer. "
        "Write a short, catchy, and fun caption with emojis when suitable."
    ),
    "tagline": (
        "You are a creative brand marketer. "
        "Write a short, powerful tagline for the given description. "
        "Keep it crisp and memorable."
    )
}

print("Choose a mode:")
print("1️⃣ Blog\n2️⃣ Caption\n3️⃣ Tagline")
choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    selected_mode = "blog"
elif choice == "2":
    selected_mode = "caption"
elif choice == "3":
    selected_mode = "tagline"
else:
    print("Invalid choice. Defaulting to 'caption'.")
    selected_mode = "caption"

system_prompt = system_prompts[selected_mode]
user_input = input("\nDescribe the image or topic: ")

template = "{system_prompt}\n\nDescription: {description}\n\nOutput:"
prompt = PromptTemplate(input_variables=["system_prompt", "description"], template=template)

llm = OllamaLLM(model="tinyllama")

chain = LLMChain(prompt=prompt, llm=llm)

result = chain.run({
    "system_prompt": system_prompt,
    "description": user_input
})

print(f"\n Mode Selected: {selected_mode.capitalize()}")
print("\n Generated Output \n")
print(result)

#pip install streamlit==1.33.0 langchain==0.2.11 langchain-core==0.2.43 langchain-ollama==0.1.3 ollama==0.3.3 numpy==1.26.4 tenacity==8.5.0 langsmith==0.1.147
