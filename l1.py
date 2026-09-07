from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Original length:", len(text))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n{c}")