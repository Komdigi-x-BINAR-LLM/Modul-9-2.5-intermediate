# Hands on modul 2.5

## Petunjuk setup
### Setup environment variables
Hands on ini memerlukan setup environment variables. Copy file .env.example pada repository ini, lalu namakan sebagai .env. Kemudian isi `API_KEY_LLM` dengan API key Gemini yang sudah anda dapatkan sebelumnya.

Jika anda menggunakan uv, environment variable dalam .env bisa langsung dimasukkan saat script dijalankan sesuai cara di bawah.

Jika anda menggunakan pip, anda perlu menginstall library tambahan:
```sh
pip install python-dotenv
```

Lalu anda perlu menambahkan ini di awal setiap script:
```py
from dotenv import load_dotenv
load_dotenv()
```

### Rute 1: Menggunakan uv
Pastikan anda sudah menginstall uv. Silahkan ikuti petunjuk instalasinya pada [link ini](https://docs.astral.sh/uv/getting-started/installation/).

Pertama, buat sebuah environment uv dengan perintah
```sh
uv init
```

Kemudian anda bisa langsung install library yang dibutuhkan:
```sh
uv add python-docx gradio openai langchain langchain-openai langchain-google-genai langchain-huggingface langchain-text-splitters langchain-chroma jinja2 deepeval
```

Untuk menjalankan sebuah script:
```sh
uv run --env-file .env /path/ke/script.py
```
### Rute 2: Menggunakan pip
Pertama, buat sebuah virtual environment  dengan perintah
```sh
python -m venv env # akan membuat virtual environment yang disimpan dalam ./env
```

Kemudian anda perlu mengaktifkan virtual environment yang sudah dibuat.
```sh
# macos/linux
./env/scripts/activate
# windows
./env/Scripts/activate.bat
```

Kemudian anda bisa langsung install library yang dibutuhkan:
```sh
pip install python-docx gradio openai langchain langchain-openai langchain-google-genai langchain-huggingface langchain-text-splitters langchain-chroma jinja2 deepeval
```

Untuk menjalankan sebuah script (pastikan sudah mengaktifkan virtual environment):
```
python /path/ke/script.py
```

## Penjelasan kode yang tersedia
1. [minimumrag.py](hands-on/minimumrag.py) - Implementasi rag sangat sederhana.
2. [chat_with_your_docs.py](hands-on/chat_with_your_docs.py) - Implementasi rag dengan ingestion dalam bentuk aplikasi *chat with your docs*
3. [deepeval_rag.py](hands-on/deepeval_rag.py) - Contoh evaluasi rag

## Tantangan
- Dari semua contoh di atas coba variasikan:
    - untuk minimumrag, coba buat dokumen sendiri (bisa pakai AI) lalu pakai untuk menggantikan dokumen yang ada, lalu uji chatbotnya
    - untuk chat-with-your-docs, sebenarnya ada tahap yang redundan. Coba cari dan ubah kodenya agar tidak redundan.
        - Hint: mungkin markdown tidak perlu disatukan dulu, setiap bagian terpisahnya bisa digunakan untuk mengelompokkan elemen-elemen menjadi chunk.
    - berdasarkan contoh dari chat-with-your-docs, buat ingestion terpisah dan persistent + aplikasi chatbot dedicated (ingestion di luar chatbot), lalu beri dokumen yang banyak. Bisa minta digenerate AI juga untuk dokumennya.
