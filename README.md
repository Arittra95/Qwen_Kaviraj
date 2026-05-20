# 🌿 কহেন কবিরাজ (Qwen Kobiraj)

A **Bengali Ayurvedic RAG Chatbot** fine-tuned on `Qwen2.5-7B-Instruct` and grounded on classical Ayurvedic texts.

> ⚠️ **Medical Disclaimer:** This project is strictly for **research, educational and fun purposes**. It is **not** a certified medical practitioner. Always consult a qualified doctor before following any advice.

## 🚀 Live Demo
#[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://#huggingface.co/spaces/YOUR_USERNAME/kobiraj-rag-chatbot)

## 🏗️ Architecture
| Component | Technology |
|-----------|------------|
| Base LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuned Model | [`Arittra/qwen-kobiraj-bengali`](https://huggingface.co/Arittra/qwen-kobiraj-bengali) |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Embeddings | `BAAI/bge-m3` |
| Vector DB | FAISS |
| UI | Gradio 4.x |

## 🛠️ Local Setup
```bash
git clone https://github.com/Arittra95/Qwen_Kaviraj.git
cd kobiraj-rag-chatbot
pip install -r requirements.txt
python app.py
```

## Attributions:

Datasets were collected from: 

> https://github.com/sciencewithsaucee-sudo, 
> https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset
