# 🧠 Arabic & English Q&A System with LangChain + HuggingFace

This project implements a **multilingual Question Answering (QA) system** that supports **English** and **Arabic** using two different pipelines:

- ✅ **English QA**: Built using `LangChain`, `FAISS`, and `HuggingFaceHub` (FLAN-Alpaca).
- ✅ **Arabic QA**: Built using `sentence-transformers`, `AraBERT`, and `medmediani/Arabic-KW-Model`.

---

## 🚀 Features

- 📖 Reads full-text documents (TXT files).
- 🔍 Embeds and splits documents into meaningful chunks.
- 🤖 Retrieves the most relevant context using semantic search (FAISS or cosine similarity).
- 🧠 Answers questions in **English or Arabic** depending on the pipeline.
- 🌐 Arabic NLP pipeline powered by AraBERT & KW semantic similarity model.

---

## 📁 Project Structure

```bash
.
├── main_english.py           # English QA using LangChain
├── main_arabic.py            # Arabic QA using transformers
├── filetxt.txt               # Sample English document
├── lougehrig.txt             # Sample Arabic document 
├── README.md                 # This file
├── requirements.txt          # Dependencies
