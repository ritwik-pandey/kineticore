from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

def process_document(file_path):
    pdf_loader = PyPDFLoader(file_path)
    documents = pdf_loader.load()

    # for i,doc in enumerate(documents):
        # print(f"TEXT PREVIEW:\n{doc.page_content[:200]}...\n")
        
        # # 2. Output the extracted metadata
        # print(f"METADATA:\n{doc.metadata}")
    return documents

# pdf_loader = PyPDFLoader("sample.pdf")
# process_document(pdf_loader, "PDF")

# txt_loader = TextLoader("sample.txt")
# process_document(txt_loader, "TXT")

# # 3. MD Loader
# # Make sure you have a sample.md in your directory
# md_loader = UnstructuredMarkdownLoader("sample.md")
# process_document(md_loader, "MARKDOWN")