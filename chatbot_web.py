import streamlit as st
import chromadb
import datetime
import pandas as pd
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# === 1. Cấu hình ban đầu ===
load_dotenv()  # Tải các biến môi trường từ file .env

persist_dir = "./vector_store"
openai_api_key = os.getenv("OPENAI_API_KEY")  # Đọc API key từ biến môi trường

# Khởi tạo hàm nhúng
embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-ada-002")

# Khởi tạo client và collection
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection("thu_tuc_hanh_chinh", embedding_function=embedding_function)

embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-ada-002")
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection("thu_tuc_hanh_chinh", embedding_function=embedding_function)

def search_context(question, top_k=3):
    results = collection.query(query_texts=[question], n_results=top_k)
    documents = results['documents'][0]
    return "\n\n".join(documents)

def ask_gpt(question, context):
    system_prompt = f"""Bạn là một trợ lý hành chính công. Dựa trên văn bản sau, hãy trả lời câu hỏi người dùng.
Nếu không chắc chắn, hãy nói 'Tôi không chắc về điều này. Vui lòng liên hệ bộ phận hành chính.'

Văn bản: {context}
"""
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

# === Thiết lập giao diện ===
st.set_page_config(page_title="Cổng hành chính công Võ Nhai", layout="wide")
st.image("anh1.jpg", use_container_width=True)

# === Tạo session đăng nhập cán bộ ===
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# === Giao diện chatbot ===
st.markdown("### 💬 Chatbot hành chính công huyện Võ Nhai ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Nhập câu hỏi về thủ tục hành chính...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm câu trả lời..."):
            try:
                context = search_context(question)
                answer = ask_gpt(question, context)
            except Exception:
                answer = "Có lỗi khi truy xuất dữ liệu hoặc gọi GPT. Vui lòng thử lại sau."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# === PHẦN GÓP Ý THỦ TỤC CHƯA CÓ ===
st.markdown("---")
st.markdown("### 📝 Góp ý nếu không tìm thấy thủ tục hành chính")

with st.form("gopy_form"):
    ten_nguoi_gopy = st.text_input("👤 Họ tên của bạn")
    sdt_nguoi_gopy = st.text_input("📱 Số điện thoại")
    noi_dung_gopy = st.text_area("✉️ Nội dung góp ý (vd: 'Tôi không tìm thấy thủ tục cấp giấy chứng nhận...')")
    gui_gopy = st.form_submit_button("📨 Gửi góp ý")

    if gui_gopy:
        if noi_dung_gopy.strip() == "":
            st.warning("❗ Nội dung góp ý không được để trống.")
        else:
            gopy_entry = {
                "Họ tên": ten_nguoi_gopy.strip(),
                "SĐT": sdt_nguoi_gopy.strip(),
                "Nội dung góp ý": noi_dung_gopy.strip(),
                "Thời gian": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            gopy_file = "gop_y.csv"
            if os.path.exists(gopy_file):
                gopy_df = pd.read_csv(gopy_file, dtype={"SĐT": str})
                gopy_df = pd.concat([gopy_df, pd.DataFrame([gopy_entry])], ignore_index=True)
            else:
                gopy_df = pd.DataFrame([gopy_entry])
            gopy_df["SĐT"] = gopy_df["SĐT"].astype(str)
            gopy_df.to_csv(gopy_file, index=False, encoding="utf-8")
            st.success("✅ Góp ý của bạn đã được gửi tới quản trị viên. Xin cảm ơn!")

# === Đặt lịch hẹn ===
st.markdown("---")
st.markdown("### 📅 Đặt lịch hẹn làm thủ tục hành chính")

with st.form("booking_form"):
    ho_ten = st.text_input("Họ và tên")
    sdt = st.text_input("Số điện thoại")
    xa_phuong = st.selectbox("Xã/Phường", [
        "Bình Long", "Cúc Đường", "Dân Tiến", "TT Đình Cả", "La Hiên", "Lâu Thượng",
        "Liên Minh", "Nghinh Tường", "Phú Thượng", "Phương Giao", "Sảng Mộc",
        "Thần Sa", "Thượng Nung", "Tràng Xá", "Vũ Chấn", "Khác"
    ])
    thu_tuc = st.text_input("Thủ tục cần thực hiện (nhập cụ thể)")
    ngay_hen = st.date_input("Ngày hẹn", min_value=datetime.date.today())
    gio_hen = st.time_input("Giờ hẹn", value=datetime.time(hour=8, minute=0))
    submitted = st.form_submit_button("📝 Đặt lịch")

    if submitted:
        new_entry = {
            "Họ tên": ho_ten.strip(),
            "SĐT": sdt.strip(),
            "Thủ tục": thu_tuc.strip(),
            "Xã/Phường": xa_phuong,
            "Ngày hẹn": ngay_hen.strftime("%Y-%m-%d"),
            "Giờ hẹn": gio_hen.strftime("%H:%M")
        }
        file_path = "lich_hen.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype={"SĐT": str})
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])
        df["SĐT"] = df["SĐT"].astype(str)
        df.to_csv(file_path, index=False, encoding="utf-8")
        st.success("✅ Bạn đã đặt lịch thành công!")

# === Tra cứu lịch hẹn ===
st.markdown("---")
st.markdown("### 🔍 Tra cứu lịch hẹn theo số điện thoại")

file_path = "lich_hen.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path, dtype={"SĐT": str})
    sdt_tim_kiem = st.text_input("📱 Nhập số điện thoại để tra cứu")
    if sdt_tim_kiem:
        ket_qua = df[df["SĐT"].str.contains(sdt_tim_kiem.strip(), na=False)]
        if not ket_qua.empty:
            st.success(f"✅ Tìm thấy {len(ket_qua)} lịch hẹn:")
            st.dataframe(ket_qua)
        else:
            st.warning("❌ Không tìm thấy lịch hẹn.")
else:
    st.warning("⚠️ Hiện chưa có lịch hẹn nào.")

# === ĐĂNG NHẬP CÁN BỘ ===
st.markdown("---")
st.markdown("### 🔐 Đăng nhập cán bộ để quản lý lịch hẹn")

if not st.session_state.is_admin:
    with st.form("login_form"):
        user = st.text_input("Tên đăng nhập")
        pw = st.text_input("Mật khẩu", type="password")
        login_btn = st.form_submit_button("Đăng nhập")

        if login_btn:
            if user == "admin" and pw == "1234":
                st.session_state.is_admin = True
                st.success("✅ Đăng nhập thành công!")
            else:
                st.error("❌ Sai tài khoản hoặc mật khẩu.")

# === QUẢN LÝ LỊCH HẸN CHO CÁN BỘ ===
if st.session_state.is_admin and os.path.exists(file_path):
    st.markdown("### 📋 Danh sách lịch hẹn (Cán bộ quản lý)")

    df = pd.read_csv(file_path, dtype={"SĐT": str})
    df = df.reset_index(drop=True)


    selected = st.selectbox("Chọn lịch hẹn để sửa hoặc xoá", df.index,
                            format_func=lambda
                                i: f"{df['Họ tên'][i]} - {df['SĐT'][i]} - {df['Ngày hẹn'][i]} {df.get('Giờ hẹn', [''])[i]}")

    with st.form("edit_form"):
        ho_ten_new = st.text_input("Họ tên", df.at[selected, "Họ tên"])
        sdt_new = st.text_input("SĐT", df.at[selected, "SĐT"])
        thu_tuc_new = st.text_input("Thủ tục", df.at[selected, "Thủ tục"])
        xa_phuong_new = st.selectbox("Xã/Phường", df["Xã/Phường"].unique(), index=0)
        ngay_hen_new = st.date_input("Ngày hẹn",
                                     datetime.datetime.strptime(df.at[selected, "Ngày hẹn"], "%Y-%m-%d").date())

        # --- Sửa lỗi tại đây ---
        gio_hen_current = df.at[selected, "Giờ hẹn"] if "Giờ hẹn" in df.columns else "08:00"
        if not isinstance(gio_hen_current, str):
            gio_hen_current = "08:00"
        gio_hen_new = st.time_input("Giờ hẹn", datetime.datetime.strptime(gio_hen_current, "%H:%M").time())

        save_btn = st.form_submit_button("💾 Lưu thay đổi")
        delete_btn = st.form_submit_button("🗑️ Xoá lịch hẹn")

        if save_btn:
            df.at[selected, "Họ tên"] = ho_ten_new
            df.at[selected, "SĐT"] = sdt_new
            df.at[selected, "Thủ tục"] = thu_tuc_new
            df.at[selected, "Xã/Phường"] = xa_phuong_new
            df.at[selected, "Ngày hẹn"] = ngay_hen_new.strftime("%Y-%m-%d")
            df.at[selected, "Giờ hẹn"] = gio_hen_new.strftime("%H:%M")
            df.to_csv(file_path, index=False, encoding="utf-8")
            st.success("✅ Đã cập nhật lịch hẹn.")
        if delete_btn:
            df.drop(index=selected, inplace=True)
            df.to_csv(file_path, index=False, encoding="utf-8")
            st.success("🗑️ Đã xoá lịch hẹn.")

# === XEM GÓP Ý TỪ NGƯỜI DÂN (chỉ dành cho cán bộ) ===
if st.session_state.is_admin:
    st.markdown("### 📬 Danh sách góp ý từ người dân")

    gopy_file = "gop_y.csv"
    if os.path.exists(gopy_file):
        gopy_df = pd.read_csv(gopy_file, dtype={"SĐT": str})

        # Thêm cột trạng thái nếu chưa có
        if "Trạng thái" not in gopy_df.columns:
            gopy_df["Trạng thái"] = "Chưa giải quyết"
            gopy_df.to_csv(gopy_file, index=False, encoding="utf-8")

        selected_row = st.selectbox("📌 Chọn góp ý để thao tác:", gopy_df.index, format_func=lambda x: f"{gopy_df.loc[x, 'Họ tên']} - {gopy_df.loc[x, 'Nội dung góp ý'][:30]}...")

        selected_gopy = gopy_df.loc[selected_row]
        st.write("### ✏️ Chi tiết góp ý đã chọn:")
        st.write(selected_gopy)

        # Sửa nội dung
        new_noi_dung = st.text_area("📝 Sửa nội dung góp ý:", selected_gopy["Nội dung góp ý"])
        new_trang_thai = st.selectbox("📋 Trạng thái:", ["Chưa giải quyết", "Đã giải quyết"], index=0 if selected_gopy["Trạng thái"] == "Chưa giải quyết" else 1)

        if st.button("💾 Lưu chỉnh sửa"):
            gopy_df.at[selected_row, "Nội dung góp ý"] = new_noi_dung.strip()
            gopy_df.at[selected_row, "Trạng thái"] = new_trang_thai
            gopy_df.to_csv(gopy_file, index=False, encoding="utf-8")
            st.success("✅ Đã lưu thay đổi!")

        # Xoá góp ý
        if st.button("🗑️ Xoá góp ý này"):
            gopy_df.drop(index=selected_row, inplace=True)
            gopy_df.to_csv(gopy_file, index=False, encoding="utf-8")
            st.warning("🗑️ Góp ý đã bị xoá.")
            st.rerun()  # Đổi từ experimental_rerun thành rerun

        # Hiển thị toàn bộ góp ý
        st.markdown("### 📄 Danh sách góp ý (toàn bộ):")
        st.dataframe(gopy_df)
    else:
        st.info("Hiện chưa có góp ý nào.")


# === FOOTER THÔNG TIN UBND HUYỆN VÕ NHAI ===
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: gray; padding-top: 10px;">
        <b>CỔNG THÔNG TIN ĐIỆN TỬ HUYỆN VÕ NHAI</b><br>
        Địa chỉ: Tổ dân phố Trung Tâm, thị trấn Đình Cả, huyện Võ Nhai, tỉnh Thái Nguyên<br>
        Điện thoại: (0208) 3 871 143 | Email: ubvnh@thainguyen.gov.vn<br>
        Bản quyền: <b>Dương Hoàng Vũ</b> - Sinh viên Đại học Kỹ thuật Công nghiệp Thái Nguyên<br>
        Email: <a href="mailto:duonghoangvu283@gmail.com">duonghoangvu283@gmail.com</a><br>
        SĐT: <a href="tel:0817933332">0817 933 332</a>
    </div>
    """,
    unsafe_allow_html=True
)
