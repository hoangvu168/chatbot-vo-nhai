import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

openai.api_key = "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"  # Thay bằng API KEY

# 1. Khởi tạo Chroma
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./vector_store"))
collection = client.get_or_create_collection("thu_tuc_hanh_chinh", embedding_function=embedding_function)

# 2. Nhập câu hỏi người dùng
question = input("Người dùng hỏi: ")

# 3. Tìm các đoạn văn liên quan nhất
results = collection.query(
    query_texts=[question],
    n_results=3
)

# 4. Kết hợp các đoạn liên quan để gửi GPT
context = "\n\n".join(results["documents"][0])

prompt = f"""Bạn là trợ lý hành chính công của huyện Võ Nhai. Trả lời câu hỏi dưới đây dựa trên thông tin được cung cấp từ các tài liệu hành chính:

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời chi tiết và đầy đủ nhất:"""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

print("\n🤖 Trợ lý trả lời:\n")
print(response.choices[0].message.content)
