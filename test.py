import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Load the environment variables from the .env file
load_dotenv()

# Optional: Verify the key loaded correctly (Do not print the whole key in production!)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# 2. Initialize the OpenAI Chat Model
# LangChain automatically fetches os.getenv("OPENAI_API_KEY") behind the scenes.
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Choose your desired model
    temperature=0.7       # Controls the randomness of responses
)

# 3. Connect and query the model
try:
    response = llm.invoke("What are the benefits of using LangChain in one line?")
    print("AI Response:\n")
    print(response.content)
except Exception as e:
    print(f"An error occurred: {e}")
