# app.py
# Streamlit web demo for "Dự đoán giá nhà TP.HCM" — Nhóm 20
# Chạy: streamlit run demo/app.py

import os
import sys
import unicodedata
import math
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import glob

# ==========================================
# THIẾT LẬP ĐƯỜNG DẪN VÀ SYSTEM PATH (QUAN TRỌNG)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------
# TỪ ĐIỂN TỌA ĐỘ (QUẬN -> PHƯỜNG)
# Dùng để nội suy chính xác Tọa độ và Khoảng cách Haversine
# ---------------------------
LOCATION_DATA = {
    "Quận 1": {"Phường Bến Nghé": (10.7769, 106.7032), "Phường Bến Thành": (10.7724, 106.6981), "Phường Đa Kao": (10.7894, 106.6975), "Phường khác (Q1)": (10.7756, 106.7019)},
    "Quận 2": {"Phường Thảo Điền": (10.8062, 106.7371), "Phường An Phú": (10.8016, 106.7480), "Phường Thủ Thiêm": (10.7735, 106.7168), "Phường khác (Q2)": (10.7872, 106.7496)},
    "Quận 3": {"Phường Võ Thị Sáu": (10.7843, 106.6816), "Phường 14": (10.7885, 106.6811), "Phường khác (Q3)": (10.7843, 106.6816)},
    "Quận 4": {"Phường 1": (10.7588, 106.7014), "Phường 13": (10.7631, 106.7047), "Phường khác (Q4)": (10.7588, 106.7014)},
    "Quận 5": {"Phường 4": (10.7592, 106.6713), "Phường 11": (10.7550, 106.6625), "Phường khác (Q5)": (10.7540, 106.6635)},
    "Quận 6": {"Phường 1": (10.7480, 106.6341), "Phường 10": (10.7385, 106.6321), "Phường khác (Q6)": (10.7480, 106.6341)},
    "Quận 7": {"Phường Tân Phong (Phú Mỹ Hưng)": (10.7313, 106.7121), "Phường Tân Thuận Đông": (10.7501, 106.7410), "Phường khác (Q7)": (10.7339, 106.7262)},
    "Quận 8": {"Phường 4": (10.7371, 106.6738), "Phường 15": (10.7200, 106.6310), "Phường khác (Q8)": (10.7241, 106.6231)},
    "Quận 9": {"Phường Long Bình": (10.8427, 106.8285), "Phường Phú Hữu": (10.7961, 106.7937), "Phường khác (Q9)": (10.8277, 106.8049)},
    "Quận 10": {"Phường 12": (10.7743, 106.6669), "Phường 14": (10.7711, 106.6578), "Phường khác (Q10)": (10.7743, 106.6669)},
    "Quận 11": {"Phường 3": (10.7661, 106.6450), "Phường 15": (10.7705, 106.6551), "Phường khác (Q11)": (10.7628, 106.6433)},
    "Quận 12": {"Phường Tân Chánh Hiệp": (10.8671, 106.6413), "Phường An Phú Đông": (10.8465, 106.6905), "Phường khác (Q12)": (10.8671, 106.6413)},
    "Gò Vấp": {"Phường 10": (10.8326, 106.6653), "Phường 17": (10.8431, 106.6751), "Phường khác (Gò Vấp)": (10.8386, 106.6653)},
    "Bình Thạnh": {"Phường 22 (Vinhomes)": (10.7946, 106.7214), "Phường 25": (10.8035, 106.7150), "Phường khác (Bình Thạnh)": (10.8105, 106.7091)},
    "Tân Bình": {"Phường 2 (Sân bay)": (10.8115, 106.6644), "Phường 13": (10.8014, 106.6410), "Phường khác (Tân Bình)": (10.8014, 106.6525)},
    "Bình Tân": {"Phường Bình Trị Đông B": (10.7511, 106.6135), "Phường An Lạc": (10.7255, 106.6061), "Phường khác (Bình Tân)": (10.7652, 106.6038)},
    "Phú Nhuận": {"Phường 1": (10.7951, 106.6858), "Phường 9": (10.8031, 106.6751), "Phường khác (Phú Nhuận)": (10.7991, 106.6788)},
    "Thủ Đức": {"Phường Linh Chiểu": (10.8521, 106.7585), "Phường Hiệp Bình Chánh": (10.8265, 106.7258), "Phường khác (Thủ Đức)": (10.8494, 106.7537)},
    "Huyện Nhà Bè": {"Xã Phước Kiển": (10.6975, 106.7121), "Thị trấn Nhà Bè": (10.6805, 106.7351), "Xã khác (Nhà Bè)": (10.6952, 106.7435)},
    "Huyện Củ Chi": {"Thị trấn Củ Chi": (10.9755, 106.4951), "Xã Tân Thạnh Đông": (10.9411, 106.5821), "Xã khác (Củ Chi)": (11.0066, 106.5050)},
    "Huyện Hóc Môn": {"Thị trấn Hóc Môn": (10.8841, 106.5925), "Xã Bà Điểm": (10.8351, 106.5985), "Xã khác (Hóc Môn)": (10.8841, 106.5925)},
    "Huyện Cần Giờ": {"Thị trấn Cần Thạnh": (10.4075, 106.9601), "Xã Bình Khánh": (10.6651, 106.7621), "Xã khác (Cần Giờ)": (10.4542, 106.8732)},
    "Bình Chánh": {"Xã Bình Hưng": (10.7255, 106.6681), "Thị trấn Tân Túc": (10.6811, 106.5901), "Xã khác (Bình Chánh)": (10.7091, 106.5516)}
}

# ---------------------------
# Helpers
# ---------------------------
def format_price_triệu(price_triệu: float) -> str:
    if price_triệu >= 1000:
        price_tỷ = price_triệu / 1000.0
        return f"{price_tỷ:,.2f} Tỷ VNĐ"
    else:
        return f"{price_triệu:,.0f} Triệu VNĐ"

def calculate_haversine_distance(lat1, lon1, lat2=10.7756, lon2=106.7019):
    """Tính khoảng cách tuyệt đối đến trung tâm Quận 1"""
    R = 6371.0
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------------------
# Caching resources (models & template)
# ---------------------------
@st.cache_resource
def load_model_from_disk() -> Optional[object]:
    candidates = [
        os.path.join(RESULTS_DIR, "tuned_lightgbm.pkl"),
        os.path.join(RESULTS_DIR, "tuned_random_forest.pkl"),
        os.path.join(RESULTS_DIR, "tuned_xgboost.pkl"),
    ]
    all_pkls = glob.glob(os.path.join(RESULTS_DIR, "*.pkl"))
    for pkl in all_pkls:
        if pkl not in candidates: candidates.append(pkl)

    errors = []
    for p in candidates:
        if os.path.exists(p):
            try:
                loaded_obj = joblib.load(p)
                if isinstance(loaded_obj, dict):
                    for key, val in loaded_obj.items():
                        if hasattr(val, 'predict'): return val
                return loaded_obj
            except Exception as e:
                errors.append(f"Lỗi đọc file {os.path.basename(p)}: {e}")
                
    for err in errors: st.sidebar.error(f"⚠️ {err}")
    return None

@st.cache_resource
def load_model_path(path: str):
    if path is None: return None
    try:
        if os.path.exists(path): 
            loaded_obj = joblib.load(path)
            if isinstance(loaded_obj, dict):
                for key, val in loaded_obj.items():
                    if hasattr(val, 'predict'): return val
                st.sidebar.error(f"⚠️ File chứa dictionary nhưng không có model nào.")
            return loaded_obj
    except Exception as e:
        st.sidebar.error(f"⚠️ Lỗi giải mã file {os.path.basename(path)}: {e}")
        return None

@st.cache_resource
def load_preprocessor():
    path = os.path.join(RESULTS_DIR, "preprocessor.pkl")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠️ File preprocessor.pkl tồn tại nhưng bị lỗi giải mã: {e}")
            return None
    return None

# ---------------------------
# Build UI
# ---------------------------
st.set_page_config(page_title="Demo Dự đoán Giá Nhà TP.HCM — Nhóm 20", layout="wide")
st.title("Demo: Dự đoán Giá Nhà TP.HCM (Nhóm 20)")
st.write("Ứng dụng demo: Nhập thông tin bất động sản, hệ thống sẽ sử dụng mô hình AI kết hợp quy tắc thị trường để dự đoán giá trị BĐS và sự chênh lệch giá giữa Chính chủ và Môi giới.")

st.sidebar.header("Thông tin bất động sản")

# 1. Nạp Model
model_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.pkl')))
model_options = ['Auto'] + [os.path.basename(p) for p in model_files if "preprocessor" not in p]
selected_model_file = st.sidebar.selectbox("Chọn mô hình dự đoán", model_options, index=0)

model = load_model_from_disk() if selected_model_file == 'Auto' else load_model_path(os.path.join(RESULTS_DIR, selected_model_file))
preprocessor = load_preprocessor()

# Trạng thái hệ thống
if model is None or preprocessor is None:
    st.sidebar.error("❌ Thiếu file Model hoặc Preprocessor trong thư mục results/")
else:
    st.sidebar.success("✅ Hệ thống AI đã sẵn sàng.")
st.sidebar.markdown("---")

# 2. Thu thập dữ liệu Vị trí (Quận -> Phường)
district = st.sidebar.selectbox("Chọn Quận/Huyện", list(LOCATION_DATA.keys()), index=0)
ward_options = list(LOCATION_DATA[district].keys())
ward = st.sidebar.selectbox("Chọn Phường/Xã", ward_options, index=0)

# Lấy tọa độ dựa trên Phường đã chọn
selected_lat, selected_lon = LOCATION_DATA[district][ward]

# 3. Thu thập dữ liệu Vật lý
area = st.sidebar.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=60.0, step=1.0)
width = st.sidebar.number_input("Chiều ngang mặt tiền (m)", min_value=1.0, max_value=100.0, value=4.0, step=0.1)

bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=20, value=2, step=1)
bathrooms = st.sidebar.number_input("Số phòng tắm", min_value=0, max_value=20, value=1, step=1)
floors = st.sidebar.number_input("Số lầu (tầng)", min_value=0, max_value=50, value=1, step=1)

# 4. Thu thập dữ liệu Pháp lý & Loại hình
has_so_hong = st.sidebar.selectbox("Pháp lý", ["Sổ hồng", "Không rõ / Chưa có", "Sổ đỏ"])
property_type = st.sidebar.selectbox(
    "Loại BĐS", 
    ["Nhà riêng", "Chung cư", "Đất", "Kho, nhà xưởng", "Khách sạn", "Nhà trọ", "Văn phòng", "Khác"]
)
direction = st.sidebar.selectbox("Hướng nhà", ["Không rõ", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"]) 

position = st.sidebar.selectbox("Vị trí (position)", ["Không rõ", "Đường chính/Mặt tiền", "Hẻm"]) 
if position == "Hẻm":
    alley_width = st.sidebar.number_input("Độ rộng hẻm (m)", min_value=1.0, max_value=12.0, value=4.0, step=0.5)
    st.sidebar.caption("💡 Hẻm tại TP.HCM thường <= 12m.")
else:
    alley_width = 0.0

# ==========================================
# KHỐI XỬ LÝ DỮ LIỆU THÔ VÀ DỰ ĐOÁN
# ==========================================
def build_raw_dataframe() -> pd.DataFrame:
    """Tạo DataFrame thô, tính toán chiều dài và khoảng cách tự động"""
    # Xử lý Tên Loại BĐS
    prop_type_val = "Căn hộ chung cư" if property_type == "Chung cư" else property_type

    # Suy luận Loại đường & Vị trí
    if position == "Đường chính/Mặt tiền":
        pos_val, auto_road_type = "Đường chính", "Đường nhựa"
    elif position == "Hẻm":
        pos_val, auto_road_type = "Trong hẻm", "Đường bê tông"
    else:
        pos_val, auto_road_type = "Unknown", "Unknown"

    # Xử lý Chiều dài (Length) nội suy từ Diện tích và Chiều ngang
    calculated_length = float(area) / float(width) if float(width) > 0 else 0.0

    # Tính Haversine
    dist_to_center = calculate_haversine_distance(selected_lat, selected_lon)

    data = {
        'Area': [float(area)],
        'Width': [float(width)],  
        'Length': [calculated_length], 
        'Bedrooms': [float(bedrooms)],
        'Bathrooms': [float(bathrooms)],
        'Floors': [float(floors)],
        'Alley Width': [float(alley_width)],
        'Agent Listing Count': [0.0], # <--- TRẢ LẠI CỘT NÀY LÀM DỮ LIỆU GIẢ ĐỂ BYPASS LỖI
        'Latitude': [selected_lat], 
        'Longitude': [selected_lon],
        'distance_to_center': [dist_to_center],
        'Direction': [direction if direction != "Không rõ" else "Unknown"],
        'Road Type': [auto_road_type],
        'Position': [pos_val],
        'Property Type': [prop_type_val],
        'District': [district]
    }
    return pd.DataFrame(data)

st.header("Dữ liệu đầu vào")
st.info("Mô hình được sử dụng cho mục đích học tập và minh họa trong phạm vi Đại học. Kết quả dự đoán chỉ mang tính chất tham khảo và cần sự tham khảo từ chuyên gia, không nên sử dụng cho quyết định đầu tư thực tế.")

# Lấy một bản mẫu để hiển thị cho người dùng (Ẩn bớt các cột kỹ thuật để giao diện sạch)
raw_df_display = build_raw_dataframe()
cols_to_hide = ['Length', 'Latitude', 'Longitude', 'distance_to_center']
display_df = raw_df_display.drop(columns=[c for c in cols_to_hide if c in raw_df_display.columns])

st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.header("Chính chủ vs Môi giới")

if st.button("🚀 Dự đoán ngay!!", type="primary"):
    if model is None or preprocessor is None:
        st.error("❌ Không thể dự đoán. Hệ thống thiếu File Model hoặc Preprocessor.")
    else:
        try:
            # 1. TẠO DỮ LIỆU CỐT LÕI (Không liên quan đến Môi giới)
            df_input = build_raw_dataframe()

            # 2. QUA PREPROCESSOR
            X_transformed = preprocessor.transform(df_input)

            # Ép kiểu dữ liệu nếu mô hình không hỗ trợ Sparse Matrix
            import scipy.sparse
            if scipy.sparse.issparse(X_transformed):
                X_transformed = X_transformed.toarray()

            # 3. DỰ ĐOÁN BẰNG AI (CHỈ 1 LẦN) ĐỂ LẤY GIÁ GỐC
            pred_log = model.predict(X_transformed)
            gia_chinh_chu = float(np.expm1(pred_log)[0])

            # 4. BUSINESS RULE (Quy tắc ngành): CỘNG 2% HOA HỒNG MÔI GIỚI
            ty_le_hoa_hong = 0.02
            chenh_lech = gia_chinh_chu * ty_le_hoa_hong
            gia_moi_gioi = gia_chinh_chu + chenh_lech
            phan_tram_chenh = ty_le_hoa_hong * 100

            # HIỂN THỊ KẾT QUẢ TRỰC QUAN
            st.success("✅ Hệ thống đã phân tích thành công. Dưới đây là kết quả dự đoán giá trị BĐS khi giao dịch chính chủ và qua môi giới.")
            
            colA, colB = st.columns(2)
            with colA:
                st.markdown(
                    f"<div style='background:#f6f9ff;padding:20px;border-radius:10px;text-align:center;border:1px solid #e1e4e8'>"
                    f"<h4 style='color:#333;margin:0'>🏡 Chính chủ</h4>"
                    f"<h2 style='color:#0b6ff0;margin:10px 0'>{format_price_triệu(gia_chinh_chu)}</h2>"
                    f"<div style='color:#666;font-size:14px'>≈ {gia_chinh_chu:,.0f} Triệu VNĐ</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            with colB:
                st.markdown(
                    f"<div style='background:#fffcf6;padding:20px;border-radius:10px;text-align:center;border:1px solid #fbe6c4'>"
                    f"<h4 style='color:#333;margin:0'>🤝 Môi giới</h4>"
                    f"<h2 style='color:#f28500;margin:10px 0'>{format_price_triệu(gia_moi_gioi)}</h2>"
                    f"<div style='color:#666;font-size:14px'>≈ {gia_moi_gioi:,.0f} Triệu VNĐ</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # BÌNH LUẬN CỦA HỆ THỐNG
            st.info(f"💡 Dựa trên việc tìm hiểu thị trường, hệ thống tính toán cộng thêm **{phan_tram_chenh:.0f}%** (tương đương chênh lệch **{chenh_lech:,.0f} Triệu VNĐ**) đại diện cho chi phí hoa hồng và marketing nếu giao dịch được thực hiện qua kênh môi giới.")

        except Exception as e:
            st.error(f"❌ Có lỗi trong luồng biến đổi dữ liệu: {e}")