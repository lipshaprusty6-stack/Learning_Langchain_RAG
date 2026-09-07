import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

response = model.invoke("What is RAG? Explain in simple words.")
print("\n--- ANSWER ---\n")
print(response.content)