# --- PATCH SQLITE (OPTIONAL) ---
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import json
import numpy as np
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
#  CẤU HÌNH CHUNG
# ============================================================

st.set_page_config(page_title="BeaBot - AI Business Agent", page_icon="🤖")
st.title("🤖 BeaBot: Trợ lý Tự động hóa Doanh nghiệp")

st.write(
    "BeaBot có thể:\n"
    "- Tra cứu **CSKH chính sách** từ nội bộ tài liệu (RAG từ policy.pdf)\n"
    "- Trả lời về **tình trạng hàng hóa** (Inventory giả lập, bảo mật kho)\n"
    "- Tư vấn chung & chuyển sang nhân viên khi cần thiết.\n"
)

# --- API KEY ---
# Lấy API key từ Streamlit Secrets (bảo mật)
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# DÙNG GEMINI CHAT (API KEY từ Google AI Studio)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    google_api_key=api_key,
    max_retries=1,
)

# ============================================================
#  ADMIN MODE (NỘI BỘ)
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Chế độ nội bộ (Quản trị viên)")

if "admin_code" not in st.session_state:
    st.session_state.admin_code = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

admin_code = st.sidebar.text_input(
    "Mã nội bộ (demo):",
    type="password",
    placeholder="vd: beabot-admin-2025",
    value=st.session_state.admin_code,
)
st.session_state.admin_code = admin_code

DEMO_ADMIN_PASS = "beabot-admin-2025"
st.session_state.is_admin = (st.session_state.admin_code == DEMO_ADMIN_PASS)

if st.session_state.is_admin:
    st.sidebar.success("✅ Bạn đang ở chế độ Quản trị viên (xem được tồn kho thật).")
    if st.sidebar.button("🔓 Thoát khỏi chế độ Admin"):
        # reset admin & rerun → lịch sử admin_only sẽ bị ẩn với khách
        st.session_state.admin_code = ""
        st.session_state.is_admin = False
        st.sidebar.info("Đã thoát chế độ Quản trị viên. Đang ở chế độ khách.")
        st.rerun()
else:
    st.sidebar.info("👀 Đang ở chế độ khách hàng (không xem số lượng thật).")

# ============================================================
#  PHẦN 1: ĐỌC INVENTORY.JSON
# ============================================================

def _load_inventory():
    try:
        with open("data/inventory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def check_inventory_public(product_name: str) -> str:
    """Dành cho khách hàng: không trả lại số lượng tồn kho cụ thể."""
    db = _load_inventory()
    if db is None:
        return (
            "⚠️ Hệ thống chưa truy cập được dữ liệu kho (inventory.json). "
            "Trong bản demo, file này mô phỏng dữ liệu kho nội bộ."
        )

    key = product_name.lower().strip()
    product = db.get(key)

    if not product:
        matches = [name for name in db.keys() if key in name or name in key]
        if matches:
            suggestion = ", ".join(matches)
            return (
                "❌ Em chưa tìm thấy đúng sản phẩm đó. "
                f"Anh/chị có đang hỏi: {suggestion} không ạ?"
            )
        return "❌ Em không tìm thấy sản phẩm này trong kho giả lập ạ."

    price = product.get("price", None)
    stock = product.get("stock", 0)
    color = product.get("color", "N/A")

    if stock <= 0:
        availability_text = (
            "hiện tại **đang tạm hết hàng**. "
            "Anh/chị có thể để lại thông tin, khi có hàng em sẽ liên hệ ngay ạ."
        )
    elif stock <= 3:
        availability_text = (
            "**còn rất ít hàng**. Nếu anh/chị ưng mẫu này thì nên đặt sớm ạ."
        )
    else:
        availability_text = (
            "**đang còn hàng sẵn** tại kho, có thể giao trong thời gian ngắn ạ."
        )

    lines = []
    lines.append(f"📦 Sản phẩm: **{product_name}**")
    if price is not None:
        lines.append(f"- Giá niêm yết: **{price:,} VNĐ**")
    if color != "N/A":
        lines.append(f"- Màu: **{color}**")
    lines.append(f"- Tình trạng kho: {availability_text}")
    lines.append(
        "\n_(Lưu ý: Hệ thống bản public chỉ hiển thị trạng thái "
        "'còn hàng / hết hàng', không hiển thị số lượng tồn chi tiết.)_"
    )
    return "\n".join(lines)


def check_inventory_admin(product_name: str) -> str:
    """Dành cho admin nội bộ: có thể xem số lượng tồn kho thực tế."""
    db = _load_inventory()
    if db is None:
        return (
            "⚠️ Không đọc được inventory.json. "
            "Vui lòng kiểm tra lại file dữ liệu kho."
        )

    key = product_name.lower().strip()
    product = db.get(key)

    if not product:
        matches = [name for name in db.keys() if key in name or name in key]
        if matches:
            suggestion = ", ".join(matches)
            return (
                "❌ Không tìm thấy đúng sản phẩm đó trong kho. "
                f"Có phải anh/chị đang muốn xem: {suggestion}?"
            )
        return "❌ Không có sản phẩm này trong dữ liệu kho."

    price = product.get("price", None)
    stock = product.get("stock", 0)
    color = product.get("color", "N/A")

    lines = []
    lines.append("🧑‍💼 **CHẾ ĐỘ ADMIN – THÔNG TIN KHO NỘI BỘ**")
    lines.append(f"📦 Sản phẩm: **{product_name}**")
    if price is not None:
        lines.append(f"- Giá niêm yết: **{price:,} VNĐ**")
    if color != "N/A":
        lines.append(f"- Màu: **{color}**")
    lines.append(f"- Số lượng tồn kho hiện tại: **{stock} sản phẩm**")

    if stock <= 0:
        lines.append("- Gợi ý: Cần nhập thêm hàng / ẩn sản phẩm khỏi website.")
    elif stock <= 3:
        lines.append("- Gợi ý: Cảnh báo tồn kho thấp, nên đặt hàng bổ sung.")
    else:
        lines.append("- Gợi ý: Tồn kho ổn, có thể chạy khuyến mãi đẩy hàng.")

    lines.append(
        "\n_(Thông tin này chỉ hiển thị cho Admin; khách hàng bên ngoài sẽ "
        "không thấy số lượng tồn kho cụ thể.)_"
    )
    return "\n".join(lines)


def list_all_products() -> str:
    """
    Liệt kê tất cả sản phẩm trong inventory.json.
    Dùng cho câu hỏi kiểu: 'bên bạn bán những sản phẩm gì', 'kể tên sản phẩm', ...
    """
    db = _load_inventory()
    if not db:
        return (
            "⚠️ Hiện hệ thống chưa tải được dữ liệu kho (inventory.json). "
            "Anh/chị vui lòng hỏi lại sau hoặc liên hệ nhân viên giúp em nhé."
        )

    lines = ["📋 Hiện bên em đang bán các sản phẩm sau:"]
    for key, product in db.items():
        # Ưu tiên field 'name' nếu có, không thì dùng key
        display_name = product.get("name", key).title()
        price = product.get("price", None)
        if price is not None:
            lines.append(f"- {display_name} – khoảng **{price:,} VNĐ**")
        else:
            lines.append(f"- {display_name}")

    lines.append(
        "\nAnh/chị quan tâm mẫu nào, em kiểm tra tồn kho & ưu đãi chi tiết giúp ạ. 😊"
    )
    return "\n".join(lines)

# ============================================================
#  PHẦN 2: RAG CHÍNH SÁCH (policy.pdf)
# ============================================================

@st.cache_resource
def setup_policy_index(api_key: str):
    try:
        loader = PyPDFLoader("data/policy.pdf")
        docs = loader.load()
    except FileNotFoundError:
        st.error(
            "⚠️ Không tìm thấy `data/policy.pdf`. "
            "Vui lòng đảm bảo file chính sách nằm đúng thư mục."
        )
        return None
    except Exception as e:
        st.error(f"⚠️ Lỗi đọc policy.pdf: {e}")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    splits = splitter.split_documents(docs)
    chunks = [d.page_content for d in splits]

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        vectors = embeddings.embed_documents(chunks)
        vectors = np.array(vectors, dtype="float32")
    except Exception as e:
        st.error(f"⚠️ Lỗi tạo embedding cho policy: {e}")
        return None

    return {"chunks": chunks, "vectors": vectors, "embeddings": embeddings}


policy_index = setup_policy_index(api_key)


def answer_policy_question(question: str) -> str:
    if policy_index is None:
        return (
            "⚠️ Hệ thống tra cứu chính sách (RAG) chưa sẵn sàng. "
            "Anh/chị vui lòng liên hệ nhân viên để được tư vấn thêm."
        )

    chunks = policy_index["chunks"]
    vectors = policy_index["vectors"]
    embeddings = policy_index["embeddings"]

    try:
        q_vec = np.array(embeddings.embed_query(question), dtype="float32")
    except Exception as e:
        return f"⚠️ Lỗi embed câu hỏi: {e}"

    vec_norms = np.linalg.norm(vectors, axis=1) + 1e-8
    q_norm = np.linalg.norm(q_vec) + 1e-8
    sims = (vectors @ q_vec) / (vec_norms * q_norm)

    top_k = 3
    top_idx = sims.argsort()[-top_k:][::-1]
    context = "\n\n".join(chunks[i] for i in top_idx)

    prompt = f"""
Bạn là BeaBot, nhân viên CSKH.

Dưới đây là một số đoạn trích từ tài liệu chính sách nội bộ:

-----------------
{context}
-----------------

Câu hỏi của khách: "{question}"

Hãy trả lời ngắn gọn, đúng với tài liệu. Nếu không đủ thông tin, hãy nói khách
liên hệ nhân viên để được tư vấn thêm.
"""
    resp = llm.invoke(prompt)
    return resp.content

# ============================================================
#  PHẦN 3: ROUTING – CHỌN TOOL TÙY NGỮ CẢNH
# ============================================================

def route_and_answer(user_question: str):
    """
    Trả về:
      - answer: nội dung trả lời
      - admin_only: True nếu đây là câu trả lời chỉ dành cho admin
    """
    q_lower = user_question.lower()

    # 1. Câu hỏi chính sách
    policy_keywords = [
        "chính sách",
        "đổi trả",
        "hoàn tiền",
        "bảo hành",
        "bị lỗi",
        "vào nước",
        "giờ làm việc",
        "mở cửa",
        "thời gian làm việc",
        "ship",
        "giao hàng",
        "phí vận chuyển",
    ]
    if any(k in q_lower for k in policy_keywords):
        return answer_policy_question(user_question), False

    # 2. Câu hỏi tồn kho / sản phẩm cụ thể
    inventory_keywords = ["iphone 15", "samsung s24", "macbook air m2"]
    for name in inventory_keywords:
        if name in q_lower:
            if st.session_state.get("is_admin", False):
                # Admin: xem số lượng thật -> admin_only
                return check_inventory_admin(name), True
            else:
                # Khách hàng: bản bảo mật
                return check_inventory_public(name), False

    # 2b. Câu hỏi liệt kê danh sách sản phẩm
    product_list_keywords = [
        "bên bạn bán những sản phẩm gì",
        "bên bạn bán gì",
        "các sản phẩm bên bạn",
        "kể tên các sản phẩm",
        "kể tên sản phẩm",
        "danh sách sản phẩm",
        "có những sản phẩm nào",
        "bạn đang bán gì",
    ]
    if any(k in q_lower for k in product_list_keywords):
        # Đây là thông tin cho khách, không phải admin-only
        return list_all_products(), False

    # 3. Human handoff
    if "gặp người" in q_lower or "nhân viên" in q_lower or "gặp quản lý" in q_lower:
        return (
            "⚠️ Vấn đề này có vẻ phức tạp. "
            "BeaBot xin phép chuyển anh/chị sang **nhân viên hỗ trợ** "
            "để được tư vấn chi tiết hơn.",
            False,
        )

    # 4. Tư vấn chung
    prompt = f"""
Bạn là BeaBot, trợ lý bán hàng của cửa hàng điện thoại / laptop.

Các sản phẩm tiêu biểu: iPhone 15, Samsung S24, Macbook Air M2
(và một số mẫu khác trong kho).

Khách hỏi: "{user_question}"

Yêu cầu:
- Trả lời thân thiện, xưng "em" – "anh/chị".
- Ngắn gọn nhưng CỤ THỂ, ưu tiên nhắc 1–3 sản phẩm ví dụ.
- Nếu câu hỏi quá chung chung, hãy hỏi lại 1 câu để làm rõ (tên sản phẩm / dòng máy / ngân sách).
"""
    resp = llm.invoke(prompt)
    return resp.content, False


# ============================================================
#  PHẦN 4: GIAO DIỆN CHAT
# ============================================================

if "messages" not in st.session_state:
    # mỗi message: {role, content, admin_only}
    st.session_state.messages = []

# Hiển thị lịch sử: nếu là admin -> xem hết; nếu là khách -> ẩn admin_only
for msg in st.session_state.messages:
    if msg.get("admin_only", False) and not st.session_state.is_admin:
        continue  # ẩn tin admin-only khi đang ở chế độ khách
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # 👉 Nếu đang ở chế độ Admin thì câu hỏi cũng được đánh dấu admin_only
    is_admin_now = st.session_state.is_admin

    # lưu câu hỏi user
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "admin_only": is_admin_now,   # <<== SỬA Ở ĐÂY
        }
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("BeaBot đang suy nghĩ..."):
            try:
                answer, admin_only_from_tool = route_and_answer(user_input)
            except Exception as e:
                answer, admin_only_from_tool = f"⚠️ Có lỗi khi xử lý: {e}", False

            # 👉 Nếu đang ở Admin thì trả lời cũng auto admin_only,
            #    còn nếu tool đã trả về admin_only=True (ví dụ xem tồn kho)
            #    thì vẫn giữ nguyên.
            admin_only_flag = admin_only_from_tool or is_admin_now

            st.markdown(answer)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "admin_only": admin_only_flag,  # <<== SỬA Ở ĐÂY
                }
            )







