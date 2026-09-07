import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1. Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Split document
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text)
print(f"Total Chunks: {len(chunks)}")

# 3. Create embeddings - USE NEW MODEL
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Create FAISS vector store
print("Creating vector store... please wait 10 sec")
vectorstore = FAISS.from_texts(chunks, embeddings)
print("Vector store created!")

# 5. Similarity search
question = "What is RAG?"
results = vectorstore.similarity_search(question, k=2)

# 6. Display retrieved chunks
print(f"\nQuestion: {question}")
print("\nRetrieved Information:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.page_content)

print("\nDone!")