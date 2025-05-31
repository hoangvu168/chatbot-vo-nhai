import os
import openai
import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ====== 1. TẢI BIẾN MÔI TRƯỜNG TỪ FILE .env ======
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

data_folder = "E:/chatbotmoi/data_txt"  # Đã xóa khoảng trắng thừa
persist_dir = "./vector_store"

# ====== 2. TẠO EMBEDDING FUNCTION ======
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

# ====== 3. KHỞI TẠO VECTOR DB MỚI ======
client = chromadb.PersistentClient(path=persist_dir)

# ====== 4. TẠO HOẶC LẤY COLLECTION ======
collection = client.get_or_create_collection(
    name="thu_tuc_hanh_chinh",
    embedding_function=embedding_function
)

# ====== 5. ĐỌC TỪNG FILE VÀ TÁCH VĂN BẢN ======
def read_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                parts = [p.strip() for p in content.split("\n\n") if p.strip()]
                for i, part in enumerate(parts):
                    documents.append({
                        "id": f"{filename}_{i}",
                        "text": part,
                        "metadata": {"source": filename}
                    })
    return documents

# ====== 6. LỌC DỮ LIỆU MỚI CHƯA CÓ ======
existing_ids = set()
try:
    existing_ids = set(collection.get()["ids"])
except:
    pass

all_docs = read_documents(data_folder)
new_docs = [doc for doc in all_docs if doc["id"] not in existing_ids]

# ====== 7. THÊM DỮ LIỆU MỚI ======
if new_docs:
    collection.add(
        documents=[doc["text"] for doc in new_docs],
        ids=[doc["id"] for doc in new_docs],
        metadatas=[doc["metadata"] for doc in new_docs]
    )
    print(f"✅ Đã thêm {len(new_docs)} đoạn văn bản mới vào vector database.")
else:
    print("⚠️ Không có dữ liệu mới để thêm.")
