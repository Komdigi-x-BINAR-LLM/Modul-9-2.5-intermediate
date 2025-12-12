from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import json
from glob import glob
import os

# ===========================================
# inisialisasi variabel penting
# ===========================================
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
DOCS_PATH = "samples/articles/"
DOCUMENTS = list()
for doc in glob(os.path.join(DOCS_PATH, "*.txt")):
    with open(doc, encoding="utf8") as f:
        DOCUMENTS.append(f.read())
print("Downloading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Downloading embedding model complete")
# ===========================================
# menghitung embedding untuk mengisi "database"
# ===========================================
BACKUP_PATH = "dataset.npy"
print("Calculating embeddings...")
if os.path.exists(BACKUP_PATH):
    DOC_EMBEDDINGS = np.load(BACKUP_PATH)
else:
    DOC_EMBEDDINGS = embedder.encode(
        DOCUMENTS, normalize_embeddings=True, batch_size=len(DOCUMENTS)
    )
    np.save(BACKUP_PATH, DOC_EMBEDDINGS)
print("Calculating embeddings complete")


# ===========================================
# implementasi retrieval sederhana pakai cosine similarity
# ===========================================
def retrieve_docs(query: str, top_k: int = 3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(DOC_EMBEDDINGS, q_emb)

    idx = sims.argsort()[::-1][:top_k]
    results = [{"document": DOCUMENTS[i], "score": float(sims[i])} for i in idx]
    results = [result for result in results if result["score"] >= SIMILARITY_THRESHOLD]
    return results


# ===========================================
# cara memberikan informasi tool yang tersedia ke LLM
# kalau menggunakan format API openAI
# ===========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "Retrieve top-k documents relevant to a query from the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = """\
# ROLE
Take the role of a football knowledge assistant. Your job is to answer the user's question or just chat.
# INSTRUCTION
You will be given a question by the user that you must answer.
You also have access to a knowledge database of football players.
When a user ask a general question that is not related to football players, give a brief answer, then tell them kindly that it's not your expertise and encourage them to ask you if they have football player related questions.
When the question is about football players or mentions a name that could be a football player, do a search using the tool given to you and answer using the information returned. If there is no information, tell the user you don't know.
# RULES
- Answer using Indonesian language.
- Answer about football players must be based only on the information given.
- Avoid referencing your database. Answer naturally as if the answer comes from you yourself. When you can't find a source, just say you don't know the answer.
"""

# ===========================================
# inisiasi LLM, definisi fungsi sederhana
# untuk bertanya ke LLM
# ===========================================
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-flash"
client = OpenAI(
    api_key=os.getenv("API_KEY_LLM"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def chat(query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # Pemanggilan pertama
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # Kalau model memanggil tool, responnya akan berisi
    # tool call
    if msg.tool_calls:
        for call in msg.tool_calls:
            print(call.to_json())
            ## logika percabangan pemrosesan buat setiap tool
            if call.function.name == "retrieve_docs":
                args = json.loads(call.function.arguments)

                # LLM cuma memberikan parameter dan menentukan tool yang dipanggil
                # implementasi toolnya dipanggil dalam kode
                results = retrieve_docs(args["query"], TOP_K)

                # Cara khusus untuk mengembalikan hasil tool call
                # ke model untuk dipakai menghasilkan jawaban
                messages.append(msg)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(results),
                    }
                )
                # lanjut menghasilkan jawaban
                final = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages
                )
                return final.choices[0].message.content

    # Kalau modelnya nggak manggil tool,
    # langsung berikan saja jawabannya
    return msg.content


# ===========================================
# Mencoba beberapa kasus
# ===========================================

## Case 1, ditanyain tentang si Rafael yang ada jawabannya
print(chat("Di klub apa Rafael Amani pertama kali bermain?"))

## Case 2, ditanyain tentang pemain bola lain yang gak ada di database
print(chat("Berapa jumlah gol yang dicetak Zinedine Zidane sepanjang karirnya?"))

## Case 3, ditanyain random aja yang gak ada hubungannya sama pemain bola
print(chat("Siapa yang membuat nasi uduk?"))
