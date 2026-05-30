# app.py
# Streamlit web demo for "Dự đoán giá nhà TP.HCM" — Nhóm 20
# Chạy: streamlit run app.py
#
# Yêu cầu: có file dữ liệu mẫu cleaned_test.csv
# và / hoặc mô hình đã lưu tại `results/stacking_model.pkl` hoặc `results/random_forest_model.pkl`.
#
# Mô tả: Tạo 1 hàng dữ liệu có đúng cấu trúc (74 cột) tương tự cleaned_test,
# build feature (Area_per_Bedroom), gọi model.predict() (log1p space) và hiển thị giá thực tế bằng np.expm1()

import os
import unicodedata
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import glob

# ---------------------------
# Helpers
# ---------------------------


def normalize_text(s: str) -> str:
    """Chuẩn hóa chuỗi: lowercase, remove diacritics, strip."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace(".", "").replace(",", "").replace("  ", " ")
    return s


def find_best_district_column(district: str, cols: List[str]) -> Optional[str]:
    """
    Tìm cột district trong template bằng cách so sánh hậu tố của cột (sau 'District_')
    với giá trị chọn của user bằng cách normalize cả hai chuỗi.
    Trả về tên cột tìm được hoặc None nếu không tìm thấy.
    """
    target_norm = normalize_text(district)
    # candidates are columns that start with District_ (case sensitive in template)
    district_cols = [c for c in cols if c.lower().startswith("district_")]
    # compare normalized suffixes
    for c in district_cols:
        suffix = c[len("District_") :]
        if normalize_text(suffix) == target_norm:
            return c
    # If exact match not found, try contains
    for c in district_cols:
        suffix = c[len("District_") :]
        if target_norm in normalize_text(suffix) or normalize_text(suffix) in target_norm:
            return c
    return None


def format_price_triệu(price_triệu: float) -> str:
    """
    Hiển thị giá: nếu >= 1000 Triệu -> show in Tỷ VNĐ, else show Triệu VNĐ.
    price_triệu: số thực (triệu VND)
    """
    if price_triệu >= 1000:
        price_tỷ = price_triệu / 1000.0
        return f"{price_tỷ:,.2f} Tỷ VNĐ"
    else:
        return f"{price_triệu:,.0f} Triệu VNĐ"


# ---------------------------
# Caching resources (models & template)
# ---------------------------

@st.cache_resource
def load_model_from_disk() -> Optional[object]:
    """
    Thử load model theo thứ tự ưu tiên:
    1) results/stacking_model.pkl
    2) results/random_forest_model.pkl
    Nếu không có, trả về None.
    """
    candidates = [
        os.path.join("results", "stacking_model.pkl"),
        os.path.join("results", "random_forest_model.pkl"),
    ]
    last_exc = None
    for p in candidates:
        abs_p = os.path.abspath(p)
        print(f"[DEBUG] Checking model path: {abs_p} (exists={os.path.exists(p)})")
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                print(f"[DEBUG] Loaded model from {abs_p}")
                return model
            except Exception as e:
                last_exc = e
                print(f"[DEBUG] Failed to load {abs_p}: {e}")
                # continue to next candidate
    # nếu không load được file nào, trả về None
    return None


@st.cache_resource
def load_template_columns(path: str = "data/processed/cleaned_test.csv") -> Optional[List[str]]:
    """
    Load một hàng mẫu từ cleaned_test để lấy danh sách cột (cấu trúc 74 cột).
    Nếu file không tồn tại hoặc lỗi đọc, trả về None.
    """
    try:
        df_template = pd.read_csv(path, nrows=5)  # chỉ cần header
        return list(df_template.columns)
    except Exception:
        return None


# ---------------------------
# Build UI
# ---------------------------

st.set_page_config(page_title="Demo Dự đoán Giá Nhà TP.HCM — Nhóm 20", layout="wide")
st.title("Demo: Dự đoán Giá Nhà TP.HCM (Nhóm 20)")
st.write("Ứng dụng demo: nhập thông tin bất động sản, nhấn 🚀 để nhận dự đoán tham khảo từ mô hình.")

# Sidebar: Input controls
st.sidebar.header("Thông tin bất động sản (Input)")

# Model selector: list .pkl files in results/
model_files = sorted(glob.glob(os.path.join('results', '*.pkl')))
model_options = ['Auto'] + [os.path.basename(p) for p in model_files]
selected_model_file = st.sidebar.selectbox("Chọn file mô hình (Auto chọn stacking/rf)", model_options, index=0)

# Whitelist Quận/Huyện (thêm/bớt theo dataset thực tế)
DISTRICT_WHITELIST = [
    "Quận 1",
    "Quận 2",
    "Quận 3",
    "Quận 4",
    "Quận 5",
    "Quận 6",
    "Quận 7",
    "Quận 8",
    "Quận 9",
    "Quận 10",
    "Quận 11",
    "Quận 12",
    "Gò Vấp",
    "Bình Thạnh",
    "Tân Bình",
    "Bình Tân",
    "Phú Nhuận",
    "Thủ Đức",
    "Huyện Nhà Bè",
    "Huyện Củ Chi",
    "Huyện Hóc Môn",
    "Huyện Cần Giờ",
    "Bình Chánh",
]
district = st.sidebar.selectbox("Chọn Quận/Huyện", DISTRICT_WHITELIST, index=0)

area = st.sidebar.number_input("Diện tích (m²)", min_value=10.0, max_value=500.0, value=60.0, step=1.0)
bedrooms = st.sidebar.number_input("Số phòng ngủ", min_value=1, max_value=10, value=2, step=1)
alley_width = st.sidebar.number_input(
    "Độ rộng hẻm (m) — nhập 0 nếu nhà mặt tiền", min_value=0.0, max_value=20.0, value=4.0, step=0.5
)
# Thêm biến phân loại phụ
has_so_hong = st.sidebar.selectbox("Pháp lý", ["Sổ hồng", "Không rõ / Chưa có", "Sổ đỏ"])
property_type = st.sidebar.selectbox("Loại BĐS (tuỳ chọn)", ["Nhà riêng", "Chung cư", "Đất", "Văn phòng", "Khác"])
# New inputs: bathrooms, direction, floors, road type, position
bathrooms = st.sidebar.number_input("Số phòng tắm", min_value=0, max_value=10, value=1, step=1)
direction = st.sidebar.selectbox("Hướng nhà", ["Không rõ", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"]) 
floors = st.sidebar.number_input("Số lầu (tầng)", min_value=0, max_value=50, value=1, step=1)
road_type = st.sidebar.selectbox("Loại đường", ["Không rõ", "Đường nhựa", "Đường bê tông", "Đường đất", "Đường đá", "Khác"]) 
position = st.sidebar.selectbox("Vị trí (position)", ["Không rõ", "Đường chính", "Hẻm", "Trong ngõ", "Góc nhà", "Mặt tiền"]) 

house_grade = st.sidebar.selectbox(
    "Cấp nhà",
    ["Không rõ", "Cấp 1", "Cấp 2", "Cấp 3", "Cấp 4"],
    index=0,
)


# Load resources

@st.cache_resource
def load_model_path(path: str):
    if path is None:
        return None
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        return None


if selected_model_file == 'Auto':
    model = load_model_from_disk()
else:
    model = load_model_path(os.path.join('results', selected_model_file))

template_cols = load_template_columns()

# Show model status
if model is None:
    st.sidebar.error(
        "Không tìm thấy file mô hình trong `results/stacking_model.pkl` hoặc `results/random_forest_model.pkl`.\n"
        "Vui lòng đặt file .pkl vào thư mục results."
    )
else:
    st.sidebar.success("Mô hình đã sẵn sàng (đã load).")

if template_cols is None:
    st.sidebar.warning(
        "Không tìm thấy cleaned_test.csv. Ứng dụng sẽ cố gắng tạo template cột mặc định."
    )
else:
    st.sidebar.info(f"Template features loaded ({len(template_cols)} cột).")
    if selected_model_file != 'Auto':
        st.sidebar.info(f"Selected model file: {selected_model_file}")

st.sidebar.markdown("---")
st.sidebar.caption("Nhóm 20 — Dự án: Dự đoán giá nhà TP.HCM")

# Main: prepare row and predict
st.header("1) Xem trước dữ liệu đầu vào được gửi vào mô hình")
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("Giá trị nhập")
    st.write("- Template columns loaded:", len(template_cols) if template_cols is not None else "None")
    # show which model files exist
    stacking_exists = os.path.exists(os.path.join('results','stacking_model.pkl'))
    rf_exists = os.path.exists(os.path.join('results','random_forest_model.pkl'))
    st.write("- Stacking model file:", "Yes" if stacking_exists else "No")
    st.write("- RandomForest model file:", "Yes" if rf_exists else "No")
    st.write(f"- Diện tích: **{area} m²**")
    st.write(f"- Số phòng ngủ: **{int(bedrooms)}**")
    st.write(f"- Độ rộng hẻm: **{alley_width} m**")
    st.write(f"- Pháp lý: **{has_so_hong}**")
    st.write(f"- Loại BĐS: **{property_type}**")
    st.write(f"- Số phòng tắm: **{int(bathrooms)}**")
    st.write(f"- Hướng nhà: **{direction}**")
    st.write(f"- Số lầu: **{int(floors)}**")
    st.write(f"- Loại đường: **{road_type}**")
    st.write(f"- Vị trí: **{position}**")
    st.write(f"- Cấp nhà: **{house_grade}**")

with col2:
    st.subheader("Ghi chú")
    st.info(
        "Mô hình được huấn luyện với target đã log1p. Ứng dụng sẽ expm1() để trả về giá thực tế (Triệu VNĐ)."
    )

# Build feature vector with the same order and columns as template
def build_feature_row(
    selected_district: str,
    area: float,
    bedrooms: int,
    alley_width: float,
    has_so_hong: str,
    property_type: str,
    bathrooms: int = 1,
    direction: str = "Không rõ",
    floors: int = 1,
    road_type: str = "Không rõ",
    position: str = "Không rõ",
    house_grade: str = "Không rõ",
    template_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Trả về DataFrame 1 dòng khớp đúng cấu trúc template_columns (nếu có).
    Nếu template_columns is None, tạo 1 template tối giản gồm:
      ['Area', 'Bedrooms', 'Area_per_Bedroom', 'Alley Width', 'Has_So_Hong', 'Property Type', plus District_...]
    """
    # create minimal template if none
    if template_columns is None:
        # tạo danh sách district columns từ whitelist
        district_cols = [f"District_{d}" for d in DISTRICT_WHITELIST]
        base_cols = ["Area", "Bedrooms", "Area_per_Bedroom", "Alley Width", "Has_So_Hong", "Property Type"]
        cols = base_cols + district_cols
        df_row = pd.DataFrame([{c: 0 for c in cols}])
    else:
        cols = template_columns.copy()
        # tạo DataFrame 1 dòng với 0 or appropriate dtypes
        df_row = pd.DataFrame([{c: 0 for c in cols}])

    # Fill numeric fields if exist
    if "Area" in df_row.columns:
        df_row.loc[0, "Area"] = float(area)
    else:
        # try some common alternative names
        for alt in ["area", "AREA", "Diện tích", "Square_Meters", "Area_m2"]:
            if alt in df_row.columns:
                df_row.loc[0, alt] = float(area)
                break

    if "Bedrooms" in df_row.columns:
        df_row.loc[0, "Bedrooms"] = int(bedrooms)
    else:
        for alt in ["Bedrooms", "Bedroom", "beds", "beds_count"]:
            if alt in df_row.columns:
                df_row.loc[0, alt] = int(bedrooms)
                break

    # Alley width common names
    alley_candidates = [c for c in df_row.columns if "alley" in c.lower() or "width" in c.lower()]
    if alley_candidates:
        df_row.loc[0, alley_candidates[0]] = float(alley_width)

    # Area per bedroom interaction
    if "Area_per_Bedroom" in df_row.columns:
        df_row.loc[0, "Area_per_Bedroom"] = float(area) / max(1, int(bedrooms))
    else:
        # if not present, try to create it if template exists (we prefer not to change column order)
        if template_columns is None:
            df_row.loc[0, "Area_per_Bedroom"] = float(area) / max(1, int(bedrooms))

    # Pháp lý -> binary column common names
    # try to find column like 'Has_So_Hong' or 'Has_So_Rong'
    legal_candidates = [c for c in df_row.columns if normalize_text("so hong") in normalize_text(c)]
    if legal_candidates:
        df_row.loc[0, legal_candidates[0]] = 1 if has_so_hong == "Sổ hồng" else 0
    else:
        # fallback: create column if we have minimal template
        if "Has_So_Hong" in df_row.columns:
            df_row.loc[0, "Has_So_Hong"] = 1 if has_so_hong == "Sổ hồng" else 0

    # Property type -> set one-hot-like column if available
    # find a column that contains property_type normalized
    prop_found = False
    for c in df_row.columns:
        if normalize_text(property_type) in normalize_text(c) and ("property" in normalize_text(c) or "type" in normalize_text(c) or "loai" in normalize_text(c)):
            df_row.loc[0, c] = 1
            prop_found = True
            break
    # If not found but exact column 'Property Type' exists, set text (some models may expect encoded)
    if not prop_found and "Property Type" in df_row.columns:
        df_row.loc[0, "Property Type"] = property_type

    # District one-hot: try to find best matching column
    if any(c.lower().startswith("district_") for c in df_row.columns):
        best_col = find_best_district_column(selected_district, df_row.columns.tolist())
        if best_col:
            # zero out all district_ columns first
            for c in df_row.columns:
                if c.lower().startswith("district_"):
                    df_row.loc[0, c] = 0
            df_row.loc[0, best_col] = 1
        else:
            # if no district column matched, try to add one if template not provided (best effort)
            if template_columns is None:
                colname = f"District_{selected_district}"
                if colname not in df_row.columns:
                    df_row[colname] = 0
                # zero others and set this to 1
                for c in df_row.columns:
                    if c.startswith("District_"):
                        df_row.loc[0, c] = 0
                df_row.loc[0, colname] = 1
    else:
        # no district columns in template, optionally create one
        colname = f"District_{selected_district}"
        if colname not in df_row.columns:
            df_row[colname] = 1
        else:
            df_row.loc[0, colname] = 1

    # final type conversions (fill NA with 0)
    df_row = df_row.fillna(0)

    # Bathrooms
    if "Bathrooms" in df_row.columns:
        df_row.loc[0, "Bathrooms"] = int(bathrooms)

    # Floors / Lầu
    if "Floors" in df_row.columns:
        df_row.loc[0, "Floors"] = int(floors)

    # Direction -> try to match a direction column
    for c in df_row.columns:
        if "direction" in c.lower() and normalize_text(direction) in normalize_text(c):
            df_row.loc[0, c] = 1
            break

    # Road type -> attempt to set matching one-hot
    for c in df_row.columns:
        if "road" in c.lower() or "road type" in c.lower() or "duong" in c.lower():
            if normalize_text(road_type) in normalize_text(c):
                df_row.loc[0, c] = 1
                break

    # Position -> attempt to set matching one-hot
    for c in df_row.columns:
        if "position" in c.lower() or "vi tri" in normalize_text(c) or "vị trí" in c:
            if normalize_text(position) in normalize_text(c):
                df_row.loc[0, c] = 1
                break

    # Cấp nhà -> map to any compatible house-grade / class column if the template exposes one.
    house_grade_candidates = [
        c
        for c in df_row.columns
        if any(
            keyword in normalize_text(c)
            for keyword in ["cap nha", "capnha", "house grade", "house class", "house_class", "grade nha"]
        )
    ]
    if house_grade_candidates:
        house_grade_order = {"Không rõ": 0, "Cấp 1": 1, "Cấp 2": 2, "Cấp 3": 3, "Cấp 4": 4}
        df_row.loc[0, house_grade_candidates[0]] = house_grade_order.get(house_grade, 0)
    elif template_columns is None:
        house_grade_order = {"Không rõ": 0, "Cấp 1": 1, "Cấp 2": 2, "Cấp 3": 3, "Cấp 4": 4}
        df_row["House_Grade"] = house_grade_order.get(house_grade, 0)

    # ensure column order as template if template provided
    if template_columns is not None:
        # Some templates may have columns not created; ensure all present
        for c in template_columns:
            if c not in df_row.columns:
                df_row[c] = 0
        df_row = df_row[template_columns]  # reorder to template order

    return df_row


def align_input_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align DataFrame `X` columns to `model` expected feature names.
    - If model has `feature_names_in_`, add missing cols (zeros), drop extras, reorder.
    - Else if model has `n_features_in_`, trim or pad numeric columns to match.
    Returns a DataFrame ready to pass to `model.predict()`.
    """
    X2 = X.copy()
    try:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            # add missing
            for c in expected:
                if c not in X2.columns:
                    X2[c] = 0
            # drop extras
            extras = [c for c in X2.columns if c not in expected]
            if extras:
                X2 = X2.drop(columns=extras)
            # reorder
            X2 = X2[expected]
            return X2
        elif hasattr(model, "n_features_in_"):
            n = int(model.n_features_in_)
            # if too many cols, drop extras on the right
            if X2.shape[1] > n:
                X2 = X2.iloc[:, :n]
            # if too few, add zero columns
            while X2.shape[1] < n:
                X2[f"pad_{X2.shape[1]}"] = 0
            return X2
    except Exception:
        # if any problem, just return X unchanged
        return X
    return X2


# Build the row
feature_row = build_feature_row(
    selected_district=district,
    area=area,
    bedrooms=int(bedrooms),
    alley_width=float(alley_width),
    has_so_hong=has_so_hong,
    property_type=property_type,
    bathrooms=int(bathrooms),
    direction=direction,
    floors=int(floors),
    road_type=road_type,
    position=position,
    house_grade=house_grade,
    template_columns=template_cols,
)

st.subheader("Dòng dữ liệu được gửi vào mô hình (preview)")
st.dataframe(feature_row.T, height=420)  # show as column-vector for readability

# Prediction block
st.markdown("---")
st.header("2) Dự đoán")

predict_button = st.button("🚀 Dự đoán giá nhà", type="primary")

if predict_button:
    # Guard: model must be loaded
    if model is None:
        st.error(
            "Lỗi: ứng dụng không tìm thấy mô hình trên đĩa. Vui lòng kiểm tra:\n"
            "- results/stacking_model.pkl hoặc\n"
            "- results/random_forest_model.pkl\n\n"
            "Ứng dụng đã không thể thực hiện dự đoán."
        )
    else:
        # Try predicting with the loaded model; if it fails, fallback to random forest
        X_pred = feature_row.copy()
        used_model_path = None
        raw_pred = None
        # First attempt: use the model that was loaded earlier (likely stacking)
        try:
            # align input to model expected features
            X_aligned = align_input_to_model(X_pred, model)
            try:
                raw_pred = model.predict(X_aligned)
            except Exception:
                raw_pred = model.predict(X_aligned.values)
            used_model_path = 'results/stacking_model.pkl' if os.path.exists(os.path.join('results','stacking_model.pkl')) else 'results/random_forest_model.pkl'
        except Exception as e_primary:
            # Primary model failed; try loading RF explicitly and predict
            try:
                rf_path = os.path.join('results', 'random_forest_model.pkl')
                if os.path.exists(rf_path):
                    rf = joblib.load(rf_path)
                    X_rf = align_input_to_model(X_pred, rf)
                    try:
                        raw_pred = rf.predict(X_rf)
                    except Exception:
                        raw_pred = rf.predict(X_rf.values)
                    used_model_path = rf_path
                else:
                    raise RuntimeError(f"Primary model failed and fallback RF not found: {e_primary}")
            except Exception as e_fallback:
                st.error(f"Có lỗi xảy ra khi dự đoán: {e_fallback}")

        if raw_pred is None:
            # prediction did not complete
            if used_model_path is None:
                st.error("Dự đoán thất bại: không có mô hình thích hợp.")
        else:
            # Convert back from log1p space
            pred_actual = np.expm1(raw_pred).ravel()
            pred_triệu = float(pred_actual[0])

            # Show which model was used
            st.success("🚀 Dự đoán hoàn tất", icon="✅")
            st.markdown(
                f"<div style='background:#f6f9ff;padding:18px;border-radius:8px;text-align:center'>"
                f"<h2 style='color:#0b6ff0;margin:0'>{format_price_triệu(pred_triệu)}</h2>"
                f"<div style='color:#333;margin-top:6px'>≈ {pred_triệu:,.0f} Triệu VNĐ</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.info(
                f"Model used: `{used_model_path}` — Lưu ý: đây là ước lượng tham khảo."
            )

            # Optional: download CSV with input + prediction
            csv = feature_row.copy()
            csv["pred_log1p"] = raw_pred.ravel()
            csv["pred_triệu"] = pred_triệu
            csv_bytes = csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Tải file input + dự đoán (.csv)",
                data=csv_bytes,
                file_name="input_with_prediction.csv",
                mime="text/csv",
            )

# Footer: quick diagnostics
st.markdown("---")
st.write("Diagnostics:")
col_a, col_b = st.columns(2)
with col_a:
    st.write("- Model loaded:" , "Yes" if model is not None else "No")
    st.write("- Template columns loaded:", len(template_cols) if template_cols is not None else "None")
with col_b:
    st.write("- Feature vector shape:", feature_row.shape)
    st.write("- Example feature names:", list(feature_row.columns[:8]))

# End of app.py