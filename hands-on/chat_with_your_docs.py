import os
import uuid
import base64

import gradio as gr
from jinja2 import Template

from docx.table import Table
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx import Document as DocxDocument

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# tiga di bawah ini sesuaikan saja sama mana yang dipakai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# untuk mapping MIMETYPE gambar
format_mapping = {"jpg": "jpeg"}


# logika ekstraksi docx
def extract_docx_elements(docx_path: str) -> list[dict]:
    doc = DocxDocument(docx_path)
    elements = []

    # pemrosesan terpisah untuk gambar. Dalam format docx, setiap gambar memiliki
    # ID masing-masing. Saat ada paragraf yang berisi gambar, paragraf tersebut
    # akan diisi komponen yang memiliki referensi ke ID ini
    # jadi kita perlu simpan dulu semua id gambar agar bisa dipetakan
    # ke bagian tertentu sebuah dokumen docx.
    image_rels = {}
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.target_ref:
            try:
                image_data = rel.target_part.blob
                image_rels[rel_id] = {
                    # data gambar kita simpan sebagai base64 string
                    # selain agar implementasinya sederhana,
                    # base64 juga biasanya digunakan untuk mengirim gambar
                    # ke API LLM.
                    "data": base64.b64encode(image_data).decode(),
                    "format": rel.target_ref.split(".")[-1],
                }
            except:
                pass

    image_counter = 0

    # Untuk dokumen docx, setiap bagiannya adalah sebuah "elemen".
    # kita melakukan handling untuk dua kasus saja:
    # tipe paragraf (teks, bisa berisi gambar) dan tabel.
    for elem_idx, element in enumerate(doc.element.body):
        # handling kalau elemennya paragraf
        if isinstance(element, CT_P):  # Paragraph
            para = Paragraph(element, doc)

            # sama seperti gambar, paragraf juga punya ID bagiannya berasal
            # logikanya, gambar yang termasuk ke suatu paragraf
            # akan memiliki ID bagian yang sama. Jadi dilakukan matching di sini
            para_images = []

            # satu paragraf terdiri dari komponen yang disebut run, yaitu barisan karakter
            # yang formattingnya sama. Intinya disini kita iterasi semua komponen paragraf
            # untuk mendapatkan ID dari bagian tersebut
            for run in para.runs:
                # Secara internal, docx sebenarnya adalah sebuah ZIP yang berisi banyak XML.
                # Intinya bagian ini mencari semua gambar dalam setiap run
                # drawing adalah komponen XML yang berisi gambar,
                # blip adalah elemen yang menjadi placeholder data
                # dari blip, kita bisa ambil ID dari data yang diperlukan.
                for drawing in run.element.findall(
                    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
                ):
                    for blip in drawing.findall(
                        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
                    ):
                        embed_id = blip.get(
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                        )
                        # di sini kita sudah mendapatkan id dari gambar dalam paragraf tersebut,
                        # kita cocokkan dengan ID semua gambar yang tersimpan
                        # kalau cocok, kita masukkan informasi gambarnya ke paragraf
                        if embed_id and embed_id in image_rels:
                            image_format = image_rels[embed_id]["format"]
                            para_images.append(
                                {
                                    "rel_id": embed_id,
                                    "image_data": image_rels[embed_id]["data"],
                                    "format": format_mapping.get(
                                        image_format, image_format
                                    ),
                                    "index": image_counter,
                                }
                            )
                            image_counter += 1

            # percabangan untuk handling bermacam kasus, karena bisa jadi paragrafnya tidak ada teks (hanya gambar), atau hanya teks, atau ada teks dan gambar
            # kalau ada gambar dalam paragraf
            if para_images:
                # kasus 1: teksnya tidak kosong. Jadi kita simnpan sebagai paragraf, tapi gambarnya jadi data tambahan.
                if para.text.strip():
                    elements.append(
                        {
                            "type": "paragraph",
                            "content": para.text,
                            "style": para.style.name,
                            "position": elem_idx,
                            # kalau ada gambar yang tersimpan dalam paragraf, disimpan di sini
                            "images": para_images,
                        }
                    )
                # kasus 2: teksnya kosong (tidak ada teks), jadi kita simpan sebagai gambar independen
                else:
                    for img in para_images:
                        elements.append(
                            {
                                "type": "image",
                                "content": img["image_data"],
                                "format": img["format"],
                                "index": img["index"],
                                "position": elem_idx,
                            }
                        )
            # kasus tiga: hanya teks
            # dalam kasus ini, para_images tidak ada isinya.
            elif para.text.strip():
                elements.append(
                    {
                        "type": "paragraph",
                        "content": para.text,
                        "style": para.style.name,
                        "position": elem_idx,
                        "images": para_images,
                    }
                )
        # handling kalau elemenyna tabel
        elif isinstance(element, CT_Tbl):  # Table
            table = Table(element, doc)
            table_data = []
            # sementara simpan dulu segalanya, tabel kita simpan per baris
            # dimana baris pertama berisi header dari tabel
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)

            elements.append(
                {
                    "type": "table",
                    "content": table_data,
                    "rows": len(table.rows),
                    "cols": len(table.columns),
                    "position": elem_idx,
                }
            )

    return elements


# kita memproses tabel menjadi teks
# ada banyak cara, di sini kita pakai yang sederhana saja
# yaitu kita buat menjadi teks terstruktur yang nested
def process_table_to_text(table_data: list) -> str:
    """Convert table to text format"""
    if not table_data or len(table_data) == 0:
        return ""

    headers = table_data[0]
    
    table_texts = ["[Table]:"]
    for i, row in enumerate(table_data[1:]):
        table_texts.append(f" Row {i}:")
        for j, cell in enumerate(row):
            table_texts.append(f"  - {headers[j]}: {str(cell).replace("\n", " ")}")

    return "\n".join(table_texts)


# kita akan mengubah data menjadi format mardown, lalu kita chunk menggunakan MarkdownHeaderSplitter
# karena setiap header punya nilai semantik (menunjukkan hierarki / topik dari bagian dokumen tertentu)
# informasi tersebut kita masukkan ke teks setiap chunk untuk memberikan informasi tambahan
# dengan harapan embeddingnya lebih representatif sehingga membantu pencarian
def add_header_metadata_to_chunk(
    chunks: list[Document], headers_list: list[str]
) -> list[Document]:
    new_chunks = []
    for chunk in chunks:
        chunk_metadata = chunk.metadata
        chunk_content = chunk.page_content
        new_chunk_contents = []
        for header in headers_list:
            if header not in chunk_metadata:
                continue
            new_chunk_contents.append(f"{header}: {chunk_metadata[header]}")
        new_chunk_contents.append(chunk_content)
        new_chunks.append(
            Document(
                page_content="\n".join(new_chunk_contents), metadata=chunk_metadata
            )
        )
    return new_chunks


# prompt utama
# umumnya system prompt untuk RAG tidak perlu memberikan summary mengenai dokumen yang ada
# karena tidak praktikal. Dalam kasus ini, karena dokumennya hanya satu, kita bisa beri summary
# dari dokumennya agar LLM paham apa isi dokumennya secara keseluruhan.
# dari sini, LLM bisa lebih tau pertanyaan mana yang perlu retrieval
SYSTEM_PROMPT = Template("""# ROLE
You are a helpful assistant.
# INSTRUCTION
The user has just uploaded a document, and the summary of the document (to give you an idea) is provided below.
Even if it fails, assume the document has already been uploaded. You already have it.
When the user asks a question, determine if its casual chat or it need search, and if so, search for information.
When in doubt, prioritize searching firsts.
Make an effort to truly understand the user's question before determining what to search.

# SUMMARY ABOUT THE DOCUMENT UPLOADED
This is the summary of the document:

{{ document_summary }}

# TONE / STYLE GUIDE
You must answer in semi-formal Indonesian with a friendly and helpful tone.
Your answer must be detailed but not just merely pasting the entire information.

# RULES
1. If after using the tool you cant find relevant information to answer, tell the user you don't know.
2. Answer as if the knowledge comes from YOU, not the document. Make no mention of the documents nor your sources.
""")

# prompt khusus untuk membuat summary
GENERATE_SUMMARY_PROMPT = Template("""Summarize the following text constructed from the first few chunks of a document.
The summary should be short, at most 2-3 sentences. The aim is to give an idea about what the document is about.
While the document may be in another language, just use english as the summary.

Here are the chunks:

{{ document_chunks }}
""")

IMAGE_CAPTION_PROMPT = "Describe this image in detail. Focus on what information it conveys in the context of a document. Be concise but thorough."


class DocumentRAGSystem:
    def __init__(self):
        # kita perlu id unik untuk setiap sesi, untuk memisahkan
        # setiap entry dalam vector database
        self.session_id = str(uuid.uuid4())

        # parameter-parameter penting
        self.top_k = 5
        # agent hanya kita inisialisasi setelah ingestion
        # harusnya tidak perlu, tapi ini kasus khusus karena
        # di aplikasi ini kita mulai dari DB yang kosong yang langsung diisi
        # begitu user upload dokumennya
        self.agent = None

        # ini backup kalau misal gagal membuat summary
        self.summary = (
            "This is a default summary. If you see this, "
            "it means the ingestion failed to generate a summary, "
            "even though the document has been uploaded successfully."
            "\n\n"
            "If you can see this, then the fallback strategy is as follows:\n"
            " - if the user asks for some information, always assume it requires searching. Use the tool to look it up.\n"
            " - if the user just chats casually or is talking about the information, just reply as needed."
        )

        # kita implementasi mekanisme history sederhana saja
        self.max_history = 5
        self.history = []

        # ada dua model utama yang anda perlukan, LLM dan model embedding.
        # API Gemini yang sebelumnya sudah anda set up memiliki model embedding.
        # ada kemungkinan besar anda akan melebihi batas penggunaan free tier
        # gemini, jadi ada beberapa kode yang dikomen di bawah yang bisa menjadi pilihan lain

        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash", google_api_key=os.getenv("API_KEY_LLM")
        # )

        # kalau menggunakan model yang dideploy secara lokal
        self.llm = ChatOpenAI(
            model="Gemma-4b:latest",
            base_url="http://127.0.0.1:8080/v1",
            api_key="asdfasdf"
        )

        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="gemini-embedding-001", google_api_key=os.getenv("API_KEY_LLM")
        # )

        # kalau menggunakan mode embedding lokal
        self.embeddings = OpenAIEmbeddings(
            model="qwen3-embeddings-0.6b:latest",
            base_url="http://127.0.0.1:8081/v1",
            api_key="asdfasdf",
        )

        # bisa juga menggunakan sentence-transformers
        # menggunakan integrasi langchain dengan sentence-transformers
        # note: ini akan memerlukan download dan
        # di awal aplikasi berjalan, perlu loading dulu agak lama

        # pilih saja model embedding apapun asal bisa multimodal
        # perlu diingat efektivitasnya bisa berbeda
        # self.embeddings = HuggingFaceEmbeddings("Qwen/Qwen3-0.6B")

        # kita perlu model untuk captioning gambar, jadi perlu llm yang multimodal
        # karena gemini multimodal, bisa kita langsung pakai saja
        # kalau anda ingin menggunakan model terpisah,
        # contohnya ada di bawah
        # sangat disarankan kalau host model sendiri
        # pilih saja yang multimodal, seperti gemma
        self.vision_llm = self.llm
        # kalau ingin host sendiri vision llm yang dedicated
        # self.vision_llm = ChatOpenAI(
        #     model="qwen2.5-vl-3b:latest",
        #     base_url="http://127.0.0.1:8082/v1",
        #     api_key="asdfasdf"
        # )

        # inisiasi chroma DB
        # kita beri session ID agar unik
        # idealnya anda perlu konfigurasi persist directory kalau ingin vectorDB nya persisten
        # (disimpan setelah aplikasi selesai untuk dipakai lagi nanti),
        # misalnya kalau ingin ingestion terpisah

        # untuk aplikasi ini tidak perlu karena sifatnya temporer saja
        self.chroma = Chroma(
            # ini nama database anda. Kalau sebelumnya sudah pernah diisi
            # dan disimpan,
            # anda tinggal pakai nama yang sama
            collection_name=f"chat-with-docs-{self.session_id}",
            embedding_function=self.embeddings,
            # line di bawah bisa di un-comment kalau perlu persistensi
            # tapi di use case ini tidak perlu
            # persist_directory="my_vector_db.db"
        )

    # setelah chunking, kita buat summary untuk dokumennya
    # ini bisa dilakukan dalam ingestion lalu disimpan dalam metadata
    # tapi dalam kasus ini karena dokumennya hanya 1
    # kita simpan saja dalam system prompt (di atas sudah dibahas)
    def generate_summary(self, chunks: list[Document], n_chunks: int = 1):
        """Generate caption for image using vision LLM"""
        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": GENERATE_SUMMARY_PROMPT.render(
                            document_chunks="\n".join(
                                [chunk.page_content for chunk in chunks[:n_chunks]]
                            )
                        ),
                    }
                ]
            )

            response = self.llm.invoke([message])
            print(response)
            if len(response.content.strip()):
                self.summary = response.content.strip()
        except Exception as e:
            print(f"[Summary generation failed: {str(e)}. Using fallback summary.]")

    # ini untuk captioning gambar
    def get_image_caption(self, image_data, image_format):
        print("Captioning image...")
        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": IMAGE_CAPTION_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_data}"
                        },
                    },
                ]
            )

            response = self.vision_llm.invoke([message])
            print(f"Caption success. Caption: {response.content}")
            return response.content
        except Exception as e:
            print(f"Captioning failed. Reason: {e}")
            return f"[Image caption generation failed: {str(e)}]"

    # setelah setiap element docx diextract,
    # setiap elemennya kita ubah menjadi markdown
    # agar bisa menggunakan chunker dari langchain
    # ini practice yang baik, daripada membuat chunker
    # untuk format berbeda, lebih baik seragamkan
    # semua dokumen menjadi format yang sama
    def map_elements_to_markdown(self, elements, progress=None):
        """Convert elements to markdown chunks with metadata"""
        chunks = []
        total = len(elements)

        for i, elem in enumerate(elements):
            # progress ini bisa diabaikan dulu,
            # hanya untuk estetika saja agar nanti
            # di interface gradio ada progress bar
            # selama dokumen diproses
            if progress:
                progress(
                    (i / total) * 0.4 + 0.3,
                    desc=f"Processing element {i + 1}/{total}...",
                )

            metadata = {
                "chunk_id": i,
                "type": elem["type"],
                "position": elem.get("position", i),
            }

            # handling untuk paragraf
            if elem["type"] == "paragraph":
                style = elem.get("style", "Normal")
                content = elem["content"]

                # pemetaan sederhana saja, teks tetap jadi teks,
                # tapi header di docx (Title, Header 1,2,3 dst)
                # kita petakan jadi header markdown
                if "Title" in style:
                    content = f"# {content}"
                elif "Heading 1" in style:
                    content = f"## {content}"
                elif "Heading 2" in style:
                    content = f"### {content}"
                elif "Heading 3" in style:
                    content = f"#### {content}"

                metadata["style"] = style

                # handling kalau ada gambar
                # dalam paragraf
                if elem.get("images"):
                    metadata["has_images"] = True
                    metadata["image_count"] = len(elem["images"])

                    captions = []

                    for img_idx, img in enumerate(elem["images"]):
                        metadata[f"image_{img_idx}_data"] = img["image_data"]
                        metadata[f"image_{img_idx}_format"] = img["format"]
                        metadata[f"image_{img_idx}_index"] = img["index"]
                        # ambil caption untuk semua gambar
                        caption = self.get_image_caption(
                            img["image_data"], img["format"]
                        )
                        captions.append(caption)

                    # struktur referensi dan caption menjadi teks, satukan dengan paragraf
                    all_captions = [
                        f"Image {img['index']}\nCaption:{caption}"
                        for img, caption in zip(elem["images"], captions)
                    ]
                    content = f"{content}\n\n[Containing Images]:" + "\n\n".join(
                        all_captions
                    )

                chunks.append({"content": content, "metadata": metadata})

            # pakai metode di atas, lengkapi dengan detail dari metadata
            elif elem["type"] == "table":
                table_text = process_table_to_text(elem["content"])
                metadata["rows"] = elem["rows"]
                metadata["cols"] = elem["cols"]

                chunks.append({"content": table_text, "metadata": metadata})

            # image individual, langsung simpan sebagai caption
            elif elem["type"] == "image":
                metadata["image_data"] = elem["content"]
                metadata["image_format"] = elem["format"]
                metadata["image_index"] = elem["index"]

                # Generate caption
                caption = self.get_image_caption(elem["content"], elem["format"])

                chunks.append(
                    {
                        "content": f"[Image {elem['index']}]:\n{caption}",
                        "metadata": metadata,
                    }
                )

        return chunks

    # buat markdown penuh dari semua komponen yang sudah dibuat sebelumnya
    def construct_markdown_document(self, elements):
        """Construct full markdown document from chunks"""
        markdown_text_lines = []
        for element in elements:
            markdown_text_lines.append(element["content"])
        return "\n".join(markdown_text_lines)
    
    # Retrieval kita implementasi sebagai tool
    # jangan lupa berikan deskripsi tool serta
    # detail mengenai parameternya
    def create_tool(self):
        @tool(response_format="content_and_artifact")
        def retrieval(query: str):
            """Use this tool to look up information when it seems like the user asks an information about the document.

            Args:
                query (str): search query for the document
            """
            retrieved_docs = self.chroma.similarity_search(query=query, k=self.top_k)
            serialized_data = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            return serialized_data, retrieved_docs
        return retrieval

    # ingestion keseluruhan
    # anda bisa abaikan progress karena hanya berhubungan dengan interface gradio
    def ingest_document(self, file_path, progress=gr.Progress()):
        progress(0, desc="Sedang ekstraksi elemen...")

        print("Sedang ekstraksi elemen...")
        # ekstraksi elemen dari docx
        elements = extract_docx_elements(file_path)

        progress(0.3, desc="Memproses tabel dan gambar...")

        print("Memproses tabel dan gambar...")
        # setiap elemen dijadikan markdown
        # handling untuk gambar dan tabel sudah termasuk di dalamnya
        chunks = self.map_elements_to_markdown(elements, progress)

        progress(0.7, desc="Mengubah format menjadi markdown...")

        print("Mengubah format menjadi markdown...")
        # menyusun dokumen yang sudah diubah menjadi markdown
        markdown_doc = self.construct_markdown_document(chunks)

        progress(0.75, desc="Melakukan chunking...")

        # Split markdown by headers
        headers_to_split_on = [
            ("#", "Document Title"),
            ("##", "Section"),
            ("###", "Subsection"),
            ("####", "Subsubsection"),
        ]

        print("Melakukan chunking...")
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        md_header_splits = markdown_splitter.split_text(markdown_doc)
        ## postprocess, informasi header bisa memberi konteks untuk setiap chunk
        ## jadi dimasukkan lagi ke dalam chunk
        md_header_splits = add_header_metadata_to_chunk(
            md_header_splits, headers_list=[hts[1] for hts in headers_to_split_on]
        )

        progress(0.85, desc="Menambahkan dokumen ke vectorDB..")

        print("Menambahkan dokumen ke vectorDB..")
        # proses perhitungan embedding sudah ditangani di dalam sini
        self.chroma.add_documents(md_header_splits)

        progress(0.95, desc="Menyiapkan asisten...")

        # generate summary, langsung tersimpan dalam instance variable
        self.generate_summary(md_header_splits, n_chunks=3)

        # buat system prompt, karena perlu dimasukkan summary
        # yang dibuat secara dinamis
        rendered_system_prompt = SYSTEM_PROMPT.render(document_summary=self.summary)
        
        retrieval_tool = self.create_tool()
        
        # inisiasi agent (sebenarnya hanya LLM dengan tool)
        self.agent = create_agent(
            self.llm,
            tools=[retrieval_tool],
            system_prompt=rendered_system_prompt,
        )
        print("Selesai.")
        progress(1.0, desc="Ingestion selesai!")

        return "Pemrosesan dokumen berhasil! Silahkan pindah ke tab chat untuk bertanya ke asisten soal dokumen anda."

    # fungsi untuk handling chat
    def chat(self, message):
        """Handle chat with document"""
        if not self.agent:
            return "Please upload a document first!"

        # format pesan user menjadi format yang digunakan agent langchain
        formatted_message = {"role": "user", "content": message}
        # cara manual untuk mendapatkan history,
        # secara manual hanya kita ambil sekian pasang
        # pertanyaan-jawaban terakhir
        session_history = self.history[-self.max_history * 2 :]
        # masukan pertanyaan ke history
        self.history.append(formatted_message)

        # secara manual kita masukkan history serta pesan dari user
        result = self.agent.invoke({"messages": [*session_history, formatted_message]})
        answer = result["messages"][-1].content

        # format jawaban, lalu simpan ke history
        formatted_answer = {"role": "assistant", "content": answer}
        self.history.append(formatted_answer)

        return answer


# inisiasi sistem RAG yang dibangun
rag_system = DocumentRAGSystem()

# Fungsi pemrosesan utama
def process_document(file):
    if file is None:
        return "Please upload a document."

    # kalau ada dokumen baru, inisiasi ulang sistemnya
    global rag_system
    rag_system = DocumentRAGSystem()

    return rag_system.ingest_document(file.name)


# cara mengirimkan pesan ke gradio
def chat_interface(message, history):
    return rag_system.chat(message)


# membangun interface sederhana dengan gradio
with gr.Blocks(title="Chat dengan dokumen anda!", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Chat dengan dokumen anda")
    gr.Markdown("Upload dokumen word (.docx), lalu tanya kami soal dokumen tersebut!")

    with gr.Tab("Upload Dokumen"):
        file_input = gr.File(
            label="Upload dokumen .docx di sini", file_types=[".docx"], type="filepath"
        )
        upload_btn = gr.Button("Proses dokumen!", variant="primary", size="lg")
        status_output = gr.Textbox(label="Status", lines=5)

        upload_btn.click(
            fn=process_document, inputs=[file_input], outputs=[status_output]
        )

    with gr.Tab("Chat Assistant"):
        chatbot = gr.ChatInterface(fn=chat_interface)
if __name__ == "__main__":
    app.launch(share=False)
