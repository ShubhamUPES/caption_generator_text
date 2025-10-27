from langchain_ollama import OllamaLLM
model = OllamaLLM(model="tinyllama")
with open("systemprompt.txt", "r") as file:
    system_prompt = file.read()
user_query = input("Enter your caption topic: ")
final_prompt = f"""{system_prompt}
User query: {user_query}

Generate exactly 3 creative and catchy options.
Each should be unique, concise, and social-media friendly.
"""
response = model.invoke(final_prompt)
print("\n Generated Captions ::")
print(response)
