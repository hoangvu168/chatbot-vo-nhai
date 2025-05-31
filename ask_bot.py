import os
import openai
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ====== 1. TẢI API KEY TỪ FILE .env ======
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ====== 2. KHỞI TẠO EMBEDDING FUNCTION VÀ CLIENT ======
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./vector_store"
))

collection = client.get_or_create_collection(
    "thu_tuc_hanh_chinh",
    embedding_function=embedding_function
)

# ====== 3. NHẬP CÂU HỎI NGƯỜI DÙNG ======
question = input("👤 Người dùng hỏi: ")

# ====== 4. TÌM KIẾM CÁC ĐOẠN VĂN LIÊN QUAN ======
results = collection.query(
    query_texts=[question],
    n_results=3
)

# ====== 5. TẠO NGỮ CẢNH TỪ KẾT QUẢ ======
context = "\n\n".join(results["documents"][0])

# ====== 6. TẠO PROMPT CHO GPT ======
prompt = f"""Bạn là trợ lý hành chính công của huyện Võ Nhai. Trả lời câu hỏi dưới đây dựa trên thông tin được cung cấp từ các tài liệu hành chính:

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời chi tiết và đầy đủ nhất:"""

# ====== 7. GỬI YÊU CẦU ĐẾN GPT ======
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

# ====== 8. HIỂN THỊ KẾT QUẢ ======
print("\n🤖 Trợ lý trả lời:\n")
print(response.choices[0].message.content)
