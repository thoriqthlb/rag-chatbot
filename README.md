# 📄 Chat with Your Document. RAG PDF Chatbot

Aplikasi chatbot berbasis RAG (Retrieval-Augmented Generation) yang memungkinkan pengguna untuk bertanya tentang isi dokumen PDF secara interaktif.

## 🚀 Demo
[Live Demo] ==> https://rag-chatbot-bvhgncsscxrr3bcfzcn3sv.streamlit.app

## 🛠️ Tech Stack
- **LLM**: Groq API (llama-3.1-8b-instant)
- **Framework**: LangChain
- **Embedding**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **UI**: Streamlit

## ⚙️ Cara Kerja
1. User upload PDF
2. Dokumen di-chunk menjadi potongan kecil
3. Setiap chunk di-embed menjadi vektor menggunakan HuggingFace
4. Vektor disimpan di ChromaDB
5. Saat user bertanya, pertanyaan di-embed dan dicocokkan dengan chunk yang relevan
6. Chunk relevan + pertanyaan dikirim ke LLM untuk menghasilkan jawaban

## 🏃 Cara Menjalankan Lokal
1. Clone repo ini
2. Buat virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Buat file `.env` dan isi: `GROQ_API_KEY=your_api_key`
5. Jalankan: `streamlit run app.py`
