from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter

loader = TextLoader("sample.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


print("⚙️ Building the BM25 Lexical Index...")

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 2  

query = "API"
print(f"\n🔍 Keyword searching for: '{query}'")

sparse_results = bm25_retriever.invoke(query)

for i, result in enumerate(sparse_results, 1):
    print(f"\n--- Top Sparse Match {i} ---")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Text: {result.page_content[:200]}...")