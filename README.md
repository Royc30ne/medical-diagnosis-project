# 🧠 Multi-Modal Medical Diagnosis System

This project is a **multi-modal diagnostic assistant** that processes:
- Patient **clinical text** (symptoms, history, lab findings)
- Chest **X-ray images**
- Retrieves similar past cases using **Elasticsearch**
- Synthesizes a diagnostic hypothesis using a **lightweight LLM**

It exposes a **FastAPI** endpoint to accept inputs and return diagnostic suggestions.

---

## 📦 Features

- 🤖 Transformer-based text encoder (Bio_ClinicalBERT)
- 🖼️ CNN-based image encoder + medical image classifier
- 🔎 Elasticsearch-based case retriever
- 🧠 Instruction-tuned LLM generator (e.g. Mistral or FLAN-T5)
- ⚡ REST API via FastAPI

---

## 🧰 Setup Instructions

### 1. Clone and Create Environment

```bash
git clone https://github.com/your-username/medical-diagnosis-project.git
cd medical-diagnosis-project

# Create conda env or use virtualenv
conda create -n med python=3.10
conda activate med

# Install dependencies
pip install -r requirements.txt
```

### 2.Create .env File

You must create a .env file in the project root with your credentials:

```
HUGGINGFACE_TOKEN=your_huggingface_token
ES_HOST=https://your-elasticsearch-url
ES_API_KEY=your_elasticsearch_api_key
```

### 3. Run the FastAPI Server


```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```