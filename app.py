cat > app.py << 'EOF'
#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import torch
import faiss
import gradio as gr
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig, AutoTokenizer, BitsAndBytesConfig,
    AutoModelForCausalLM, GenerationConfig
)

MODEL_ID = "Arittra/qwen-kobiraj-bengali"
BASE_CONFIG_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = Path("./data/raw_data")
INDEX_DIR = Path("./data/faiss_index")
INDEX_FILE = INDEX_DIR / "kobiraj.index"
CHUNKS_FILE = INDEX_DIR / "chunks.npy"
EMBED_MODEL_NAME = "BAAI/bge-m3"

SYSTEM_TEXT = (
    "আপনি একজন ঐতিহ্যবাহী ও অভিজ্ঞ আয়ুর্বেদিক কবিরাজ। আপনার কাজ হলো রোগীর বর্ণিত শারীরিক উপসর্গ ও কষ্ট "
    "মনোযোগ দিয়ে শোনা এবং প্রদত্ত আয়ুর্বেদিক তথ্যভাণ্ডার (Context) ব্যবহার করে খাঁটি, স্পষ্ট ও সহানুভূতিশীল "
    "বাংলা ভাষায় ভেষজ চিকিৎসা ও পথ্য প্রদান করা।\n\n"
    "পালনীয় নিয়মাবলী:\n"
    "১. ব্যক্তিত্ব ও ভাষা: সর্বদা শান্ত, ধৈর্যশীল ও শ্রদ্ধাপূর্ণ কবিরাজী আচরণ বজায় রাখুন। "
    "আধুনিক বাংলিশ বা অপ্রাসঙ্গিক ইংরেজি শব্দ সম্পূর্ণ পরিহার করুন।\n"
    "২. তথ্যভিত্তিক সমাধান: মনগড়া কোনো পরামর্শ না দিয়ে, শুধুমাত্র নিচের দেওয়া তথ্যভাণ্ডার (Context)-এ থাকা "
    "ডেটার ওপর ভিত্তি করে নির্দিষ্ট ভেষজ, সঠিক চিকিৎসা পদ্ধতি এবং জীবনযাত্রার সামগ্রিক পরিবর্তন সাজেস্ট করুন।\n"
    "৩. যৌক্তিক ব্যাখ্যা: রোগীকে দেওয়া প্রতিটি ভেষজ বা পথ্য কেন তার উপসর্গের জন্য কার্যকর এবং কীভাবে এটি তার "
    "শারীরিক ভারসাম্য (দোষ প্রশমন) পুনরুদ্ধার করবে, তা সংক্ষেপে বুঝিয়ে বলুন।\n"
    "৪. সতর্কতা (Disclaimer): যদি লক্ষণ জটিল মনে হয় কিংবা প্রদত্ত তথ্যভাণ্ডারে পর্যাপ্ত তথ্য না থাকে, "
    "তবে অবিলম্বে রোগীকে বিনীতভাবে সতর্ক করুন এবং সরাসরি কোনো বিশেষজ্ঞ চিকিৎসকের পরামর্শ নেওয়ার নির্দেশ দিন।\n"
    "৫. খাদ্য ও পথ্য: কখনোই বিপরীত বা দ্ব্যর্থক পরামর্শ দেবেন না। জ্বর/পিত্তের ক্ষেত্রে তৈলাক্ত, মশলাদার "
    "এবং ভাজা খাবার নিষিদ্ধ বলে উল্লেখ করুন। ভেষজ নাম কেবল বাংলা বা সংস্কৃত নামে ব্যবহার করুন, মিশ্র স্ক্রিপ্ট পরিহার করুন।"
)

DISCLAIMER = """
<div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 14px; margin-bottom: 18px; border-radius: 6px; font-family: sans-serif;">
    <strong style="color: #856404; font-size: 15px;">⚠️ গুরুত্বপূর্ণ সতর্কতা / Medical Disclaimer</strong><br>
    <span style="color: #856404; font-size: 13px;">
    এই চ্যাটবটটি কেবল <strong>গবেষণার উদ্দেশ্যে</strong> তৈরি। একে কোনো সার্টিফাইড চিকিৎসকের বিকল্প হিসেবে বিবেচনা করা যাবে না। 
    এখানে প্রদত্ত যেকোনো পরামর্শ, ভেষজ বা চিকিৎসা পদ্ধতি অনুসরণ করার আগে অবশ্যই একজন যোগ্য আয়ুর্বেদিক বা আধুনিক চিকিৎসকের সাথে পরামর্শ করুন। 
    জরুরি অবস্থায় (যেমন: বুকে তীব্র ব্যথা, শ্বাসকষ্ট, রক্তপাত ইত্যাদি) দ্রুত নিকটস্থ হাসপাতালে যোগাযোগ করুন।
    </span>
</div>
"""

print("🔄 Loading Kobiraj LLM...")
config = AutoConfig.from_pretrained(BASE_CONFIG_ID)
tokenizer = AutoTokenizer.from_pretrained(BASE_CONFIG_ID)
base_generation_config = GenerationConfig.from_pretrained(BASE_CONFIG_ID)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    generation_config=base_generation_config,
    trust_remote_code=True
)
model.eval()
print("✅ LLM loaded.")

def load_raw_docs(data_path: Path):
    docs = []
    for json_file in data_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    text = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                    docs.append({"text": text, "source": json_file.name})
            else:
                docs.append({"text": json.dumps(data, ensure_ascii=False), "source": json_file.name})
    for csv_file in data_path.glob("*.csv"):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            docs.append({"text": text, "source": csv_file.name})
    return docs

def chunk_text(text, chunk_size=512, overlap=128):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ["।", ".", "!", "?", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos != -1:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks

def build_index():
    print("📚 Building FAISS index from raw data...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    raw_docs = load_raw_docs(DATA_PATH)
    all_chunks = []
    for doc in raw_docs:
        for c in chunk_text(doc["text"]):
            all_chunks.append({"text": c, "source": doc["source"]})
    print(f"🔢 Encoding {len(all_chunks)} chunks with {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    np.save(str(CHUNKS_FILE), np.array(all_chunks, dtype=object))
    print("💾 Index saved.")
    return index, all_chunks, embed_model

def load_or_build_index():
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        print("📂 Loading existing FAISS index...")
        index = faiss.read_index(str(INDEX_FILE))
        chunks = np.load(str(CHUNKS_FILE), allow_pickle=True).tolist()
        embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
        return index, chunks, embed_model
    else:
        return build_index()

index, all_chunks, embed_model = load_or_build_index()

def retrieve_context(query, k=3):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(all_chunks):
            results.append({"text": all_chunks[idx]["text"], "source": all_chunks[idx]["source"], "score": float(score)})
    return results

def chat(message, history):
    if not message.strip():
        return "", history
    messages = [{"role": "system", "content": SYSTEM_TEXT}]
    for human, bot in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": bot})
    contexts = retrieve_context(message, k=3)
    context_str = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    user_msg = (
        f"নিচের তথ্যভাণ্ডার (Context) ব্যবহার করে উত্তর দিন:\n"
        f"[তথ্যভাণ্ডার শুরু]\n{context_str}\n[তথ্যভাণ্ডার শেষ]\n\n"
        f"রোগীর প্রশ্ন: {message}"
    )
    messages.append({"role": "user", "content": user_msg})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append([message, response])
    return "", history

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="কহেন কবিরাজ",
    css="""
        .chatbot { font-size: 16px; line-height: 1.7; }
        .input-textbox { font-size: 15px; }
        footer { visibility: hidden; }
    """
) as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 4px;'>🌿 কহেন কবিরাজ 🌿</h1>")
    gr.Markdown("<p style='text-align: center; color: #555; font-size: 15px;'>আপনার ব্যক্তিগত আয়ুর্বেদিক পরামর্শদাতা</p>")
    gr.HTML(DISCLAIMER)
    chatbot = gr.Chatbot(
        label="কথোপকথন",
        height=520,
        bubble_full_width=False,
        show_copy_button=True,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/2964/2964514.png"),
    )
    with gr.Row():
        msg = gr.Textbox(
            placeholder="আপনার শারীরিক সমস্যা বা উপসর্গ বর্ণনা করুন...",
            label="",
            show_label=False,
            scale=9,
            autofocus=True,
        )
        submit = gr.Button("পাঠান ➤", scale=1, variant="primary")
    clear = gr.Button("🗑️ কথোপকথন মুছে ফেলুন", size="sm")
    submit.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, msg], queue=False)
    gr.Markdown("""
    <div style="text-align: center; margin-top: 10px; color: #999; font-size: 11px;">
        Powered by <strong>Qwen2.5-7B-Instruct</strong> • Bengali Ayurvedic Fine-tune • RAG with FAISS
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
EOF
