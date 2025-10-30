import sys
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent, AgentType

try:
    llm = OllamaLLM(model="gemma:2b")
    
    llm.invoke("Test")
    print("Ollama and gemma:2b model successfully loaded.\n")

except Exception as e:
    print(f"--- Error initializing OllamaLLM ---")
    print(f"Error: {e}")
    print("\nPlease ensure Ollama is running and you have pulled the model:")
    print("  ollama pull gemma:2b")
    print("---------------------------------")
    sys.exit(1)


blog_template = """You are a professional blog writer.
Write an engaging short blog paragraph about the given description.
Description: {description}
Output:"""
blog_prompt = PromptTemplate(input_variables=["description"], template=blog_template)
blog_chain = LLMChain(prompt=blog_prompt, llm=llm)

caption_template = """You are an Instagram caption writer.
Write a short, catchy, and fun caption with emojis.
Description: {description}
Output:"""
caption_prompt = PromptTemplate(input_variables=["description"], template=caption_template)
caption_chain = LLMChain(prompt=caption_prompt, llm=llm)

tagline_template = """You are a creative brand marketer.
Write a short, powerful tagline.
Description: {description}
Output:"""
tagline_prompt = PromptTemplate(input_variables=["description"], template=tagline_template)
tagline_chain = LLMChain(prompt=tagline_prompt, llm=llm)


tools = [
    Tool(
        name="Blog Writer",
        func=blog_chain.run,
        description="Use this tool when the user asks for a blog post, a descriptive paragraph, or informative content about a topic."
    ),
    Tool(
        name="Instagram Caption Writer",
        func=caption_chain.run,
        description="Use this tool when the user asks for a social media caption, a fun post, or something short with emojis for a picture or idea."
    ),
    Tool(
        name="Tagline Creator",
        func=tagline_chain.run,
        description="Use this tool when the user asks for a tagline, a slogan, a motto, or a very short, powerful marketing phrase for a brand or product."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print("-------------------------------------------------")
print("Hi! I'm your social media assistant.")
print("What do you need content for? (e.g., 'a tagline for my new coffee shop')")
user_request = input("> ")

if user_request:
    print("\n...Agent is thinking (this may take a moment)...\n")
    
    try:
        result = agent.run(user_request)
        
        print("\n--- Generated Output ---\n")
        print(result)
        
    except Exception as e:
        print(f"\n--- Agent Error ---")
        print(f"The agent failed, likely due to the model's limitations.")
        print(f"Error details: {e}")
        print("\nThis often happens when the model (gemma:2b) still")
        print("struggles with the 'Thought/Action/Observation' reasoning format.")

else:
    print("No input provided. Exiting.")
    
