# 🎧 VidQuery AI (RAG Pipeline)

## 📌 Project Overview
VidQuery AI is an end-to-end Retrieval-Augmented Generation (RAG) system that converts video content into a searchable AI assistant. It processes videos into text, generates embeddings, and allows users to ask questions and receive precise answers with relevant video timestamps.

---

## 🚀 Features
- 🎥 Convert videos to audio (MP3)
- 📝 Transcribe audio using Faster-Whisper (optimized)
- ✂️ Smart chunk grouping for better context
- 🔍 Generate embeddings for semantic search
- 🤖 Retrieve relevant chunks using cosine similarity
- 💬 Answer queries using LLM (Groq + Ollama fallback)
- ⏱️ Provides video number + accurate timestamps
- 🛡️ Safe data processing using temporary folder replacement

---

## 📂 Project Structure
```
project/
│
├── videos/                 
├── audios/                 
├── jsons/                  
├── embeddings.joblib       
│
├── vid_to_mp3.py           
├── mp3_to_json.py          
├── preprocess_jsons.py     
├── process_incoming.py     
```

---

## ⚙️ Workflow

### 1️⃣ Convert Videos to MP3
```
python vid_to_mp3.py
```

### 2️⃣ Transcribe & Process Audio
```
python mp3_to_json.py
```

### 3️⃣ Generate Embeddings
```
python preprocess_jsons.py
```

### 4️⃣ Ask Questions (Inference)
```
python process_incoming.py
```

---

## 🧠 Tech Stack
- Python
- Faster-Whisper (base model, INT8 quantized)
- Ollama
- Groq API
- Pandas / NumPy
- Scikit-learn
- FFmpeg

---

## 📦 Requirements
```
pip install pandas numpy scikit-learn joblib requests faster-whisper groq
```

Install FFmpeg:
```
brew install ffmpeg
```

---

## ⚠️ Prerequisites
```
ollama run llama3.2
ollama pull bge-m3
```

---

## 💡 How It Works
1. Video → audio  
2. Audio → text  
3. Text → chunks  
4. Embeddings  
5. Query → cosine similarity search  
6. LLM generates answer  

---

## 🎯 Applications
- Course assistants  
- Lecture search  
- YouTube Q&A  
- Educational AI  

---

## 👨‍💻 Author
Shlok Bhandari
B.E. Artificial Intelligence & Data Science
