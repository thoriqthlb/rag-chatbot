from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load PDF
loader = PyPDFLoader("StrukturData.pdf")
pages = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(pages)

print(f"Jumlah halaman: {len(pages)}")
print(f"Jumlah chunks: {len(chunks)}")
print(f"\nContoh chunk pertama:\n{chunks[0].page_content}")