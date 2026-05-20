from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_data(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    # 4. Output the chunk count
    # print(f"Original documents loaded: {len(documents)}")
    # print(f"Total chunks created: {len(chunks)}\n")

    # # Verify metadata preservation on the first chunk
    # print("--- FIRST CHUNK PREVIEW ---")
    # print(f"Content: {chunks[0].page_content[:150]}...")
    # print(f"Metadata: {chunks[0].metadata}")
    return chunks