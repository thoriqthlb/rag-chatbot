from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

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

# Embed & Simpan ke Chroma
embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)

print("Selesai!")
print(f"Total chunks di vectorstore: {vectorstore._collection.count()}")