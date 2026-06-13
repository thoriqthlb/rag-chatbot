from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

load_dotenv()

# Setup
@st.cache_resource
def setup_chain():

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

    # Retrieval + LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
    Jawab pertanyaan berdasarkan konteks berikut saja.
    Konteks = {context}                                                                            
    Pertanyaan = {question}                                      
    Jawaban:                                     
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

#UI
st.title("Chat with Your Document")
st.caption("RAG PDF Chatbot. Powered by Groq + LangChain")

chain = setup_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pertanyaan := st.chat_input("Tanya sesuatu tentang dokumen..."):
    st.session_state.messages.append({"role": "user", "content": pertanyaan})
    with st.chat_message("user"):
        st.markdown(pertanyaan)

    with st.chat_message("assistant"):
        jawaban = chain.invoke(pertanyaan)
        st.markdown(jawaban)

    st.session_state.messages.append({"role": "assistant", "content": jawaban})