import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# FIX: Big chunk = less API calls
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Total Chunks: {len(chunks)}")  # Should be <15 now

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

vectorstore = FAISS.from_texts(chunks, embeddings)
print("Vector store created!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.2
)
question = "What is RAG?"
docs = retriever.invoke(question)
context = "\n".join([d.page_content for d in docs])

prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
response = llm.invoke(prompt)

# Replace your 2 print lines with this:
print("\nContext:", context)

# Gemini 3.5 returns list, so we extract it
answer = response.content
if isinstance(answer, list):
    answer = answer[0].get('text', str(answer[0])) if isinstance(answer[0], dict) else str(answer[0])

print("\nFinal Answer:", answer)