import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text)
print(f"Total Chunks: {len(chunks)}")

# FIX 1: New embedding model for 2026
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

vectorstore = FAISS.from_texts(chunks, embeddings)
print("Vector store created!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# FIX 2: For NEW API keys (created after June 2026), ONLY 3.5 works
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # Google itself told you this in error!
    google_api_key=api_key,
    temperature=0.2
)

question = "What is RAG?"
print(f"\nQuestion: {question}")
docs = retriever.invoke(question)

print("\nRetrieved Information:")
for i, doc in enumerate(docs):
    print(f"Result {i+1}: {doc.page_content}")

context = "\n".join([d.page_content for d in docs])
prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer simply:"

response = llm.invoke(prompt)
print("\n--- LLM Answer ---")
print(response.content)