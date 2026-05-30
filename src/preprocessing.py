import re
import unicodedata
from math import cos, radians

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COL = "Price"
HCMC_CENTER_LAT = 10.7769
HCMC_CENTER_LON = 106.7009
HCMC_LAT_RANGE = (10.3, 11.2)
HCMC_LON_RANGE = (106.3, 107.1)

BASE_NUMERIC_COLS = [
    "Area",
    "Width",
    "Length",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Alley Width",
    "Agent Listing Count",
    "Latitude",
    "Longitude",
    "price_per_m2",
    "distance_to_center",
]

OUTLIER_CLIP_COLS = [
    "Area",
    "Width",
    "Length",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Alley Width",
    "Agent Listing Count",
    "Latitude",
    "Longitude",
    "distance_to_center",
]

BASE_CATEGORICAL_COLS = [
    "Direction",
    "Road Type",
    "Position",
    "Property Type",
    "Agent Role",
    "District",
]

DROP_COLS = [
    "Listing ID",
    "Title",
    "Description",
    "Avatar",
    "Agent Name",
    "Last Updated",
    "Scraped At",
    "Last Updated Date",
    "Property Type Slug",
    "Location",
    "Province",
    "VIP Account",
]

VALID_DISTRICT_KEYS = {
    "quan 1": "Quận 1",
    "quan 2": "Quận 2",
    "quan 3": "Quận 3",
    "quan 4": "Quận 4",
    "quan 5": "Quận 5",
    "quan 6": "Quận 6",
    "quan 7": "Quận 7",
    "quan 8": "Quận 8",
    "quan 9": "Quận 9",
    "quan 10": "Quận 10",
    "quan 11": "Quận 11",
    "quan 12": "Quận 12",
    "quan binh thanh": "Quận Bình Thạnh",
    "quan go vap": "Quận Gò Vấp",
    "quan tan binh": "Quận Tân Bình",
    "quan tan phu": "Quận Tân Phú",
    "quan binh tan": "Quận Bình Tân",
    "quan phu nhuan": "Quận Phú Nhuận",
    "quan thu duc": "Quận Thủ Đức",
    "huyen binh chanh": "Huyện Bình Chánh",
    "huyen cu chi": "Huyện Củ Chi",
    "huyen hoc mon": "Huyện Hóc Môn",
    "huyen nha be": "Huyện Nhà Bè",
    "huyen can gio": "Huyện Cần Giờ",
}

# Phường/xã thuộc TP.HCM Mới (Thủ Đức mới) — trích từ Location string
VALID_WARD_KEYS = {
    # Phường thuộc quận cũ Quận 9 (nay thuộc TP Thủ Đức)
    "phuong long binh": "Phường Long Bình",
    "phuong long thanh my": "Phường Long Thạnh Mỹ",
    "phuong tan phu": "Phường Tân Phú",
    "phuong hiep phu": "Phường Hiệp Phú",
    "phuong tang nhon phu a": "Phường Tăng Nhơn Phú A",
    "phuong tang nhon phu b": "Phường Tăng Nhơn Phú B",
    "phuong phuoc long b": "Phường Phước Long B",
    "phuong phuoc long a": "Phường Phước Long A",
    "phuong truong thanh": "Phường Trường Thạnh",
    "phuong long phuoc": "Phường Long Phước",
    "phuong long truong": "Phường Long Trường",
    "phuong phuoc binh": "Phường Phước Bình",
    "phuong tan thanh": "Phường Tân Thạnh",
    # Phường thuộc quận cũ Quận 2
    "phuong thu thiem": "Phường Thủ Thiêm",
    "phuong an loi dong": "Phường An Lợi Đông",
    "phuong thao dien": "Phường Thảo Điền",
    "phuong an phu": "Phường An Phú",
    "phuong binh an": "Phường Bình An",
    "phuong binh trung dong": "Phường Bình Trưng Đông",
    "phuong binh trung tay": "Phường Bình Trưng Tây",
    "phuong binh khanh": "Phường Bình Khánh",
    "phuong cat lai": "Phường Cát Lái",
    "phuong giong ong to": "Phường Giồng Ông Tố",
    "phuong an khanh": "Phường An Khánh",
    "phuong phu huu": "Phường Phú Hữu",
    # Phường thuộc quận cũ Thủ Đức
    "phuong linh xuan": "Phường Linh Xuân",
    "phuong binh chieu": "Phường Bình Chiểu",
    "phuong linh trung": "Phường Linh Trung",
    "phuong tam binh": "Phường Tam Bình",
    "phuong tam phu": "Phường Tam Phú",
    "phuong hiep binh chanh": "Phường Hiệp Bình Chánh",
    "phuong hiep binh phuoc": "Phường Hiệp Bình Phước",
    "phuong linh chieu": "Phường Linh Chiểu",
    "phuong linh dong": "Phường Linh Đông",
    "phuong binh tho": "Phường Bình Thọ",
    "phuong truong tho": "Phường Trường Thọ",
}

# Tọa độ trung tâm từng phường của TP.HCM Mới (TP Thủ Đức)
KNOWN_WARD_COORDS = {
    "phuong long binh": {"Latitude": 10.8240, "Longitude": 106.8020},
    "phuong long thanh my": {"Latitude": 10.8370, "Longitude": 106.8190},
    "phuong tan phu": {"Latitude": 10.8450, "Longitude": 106.8270},
    "phuong hiep phu": {"Latitude": 10.8510, "Longitude": 106.7900},
    "phuong tang nhon phu a": {"Latitude": 10.8560, "Longitude": 106.8010},
    "phuong tang nhon phu b": {"Latitude": 10.8490, "Longitude": 106.8060},
    "phuong phuoc long b": {"Latitude": 10.8330, "Longitude": 106.7720},
    "phuong phuoc long a": {"Latitude": 10.8280, "Longitude": 106.7680},
    "phuong truong thanh": {"Latitude": 10.8150, "Longitude": 106.8280},
    "phuong long phuoc": {"Latitude": 10.8060, "Longitude": 106.8410},
    "phuong long truong": {"Latitude": 10.8010, "Longitude": 106.8340},
    "phuong phuoc binh": {"Latitude": 10.8390, "Longitude": 106.7800},
    "phuong tan thanh": {"Latitude": 10.8440, "Longitude": 106.8110},
    "phuong thu thiem": {"Latitude": 10.7880, "Longitude": 106.7270},
    "phuong an loi dong": {"Latitude": 10.7940, "Longitude": 106.7380},
    "phuong thao dien": {"Latitude": 10.8020, "Longitude": 106.7340},
    "phuong an phu": {"Latitude": 10.7960, "Longitude": 106.7460},
    "phuong binh an": {"Latitude": 10.7820, "Longitude": 106.7520},
    "phuong binh trung dong": {"Latitude": 10.7760, "Longitude": 106.7560},
    "phuong binh trung tay": {"Latitude": 10.7710, "Longitude": 106.7500},
    "phuong binh khanh": {"Latitude": 10.7700, "Longitude": 106.7420},
    "phuong cat lai": {"Latitude": 10.7670, "Longitude": 106.7630},
    "phuong giong ong to": {"Latitude": 10.7610, "Longitude": 106.7690},
    "phuong an khanh": {"Latitude": 10.7860, "Longitude": 106.7430},
    "phuong phu huu": {"Latitude": 10.7900, "Longitude": 106.7560},
    "phuong linh xuan": {"Latitude": 10.8650, "Longitude": 106.7680},
    "phuong binh chieu": {"Latitude": 10.8590, "Longitude": 106.7750},
    "phuong linh trung": {"Latitude": 10.8700, "Longitude": 106.7720},
    "phuong tam binh": {"Latitude": 10.8450, "Longitude": 106.7450},
    "phuong tam phu": {"Latitude": 10.8500, "Longitude": 106.7510},
    "phuong hiep binh chanh": {"Latitude": 10.8360, "Longitude": 106.7130},
    "phuong hiep binh phuoc": {"Latitude": 10.8310, "Longitude": 106.7170},
    "phuong linh chieu": {"Latitude": 10.8600, "Longitude": 106.7590},
    "phuong linh dong": {"Latitude": 10.8530, "Longitude": 106.7640},
    "phuong binh tho": {"Latitude": 10.8470, "Longitude": 106.7580},
    "phuong truong tho": {"Latitude": 10.8410, "Longitude": 106.7620},
}

DISTRICT_KEY_PATTERN = re.compile(
    r"\b(quan\s+\d{1,2}|quan\s+[a-z\s]+|huyen\s+[a-z\s]+)\b",
    flags=re.IGNORECASE,
)

KNOWN_DISTRICT_COORDS = {
    "quan 1": {"Latitude": 10.7757, "Longitude": 106.7004},
    "quan 2": {"Latitude": 10.7873, "Longitude": 106.7498},
    "quan 3": {"Latitude": 10.7844, "Longitude": 106.6844},
    "quan 4": {"Latitude": 10.7578, "Longitude": 106.7013},
    "quan 5": {"Latitude": 10.7540, "Longitude": 106.6634},
    "quan 6": {"Latitude": 10.7465, "Longitude": 106.6355},
    "quan 7": {"Latitude": 10.7328, "Longitude": 106.7216},
    "quan 8": {"Latitude": 10.7241, "Longitude": 106.6286},
    "quan 9": {"Latitude": 10.8428, "Longitude": 106.8287},
    "quan 10": {"Latitude": 10.7732, "Longitude": 106.6678},
    "quan 11": {"Latitude": 10.7629, "Longitude": 106.6501},
    "quan 12": {"Latitude": 10.8672, "Longitude": 106.6413},
    "quan binh thanh": {"Latitude": 10.8106, "Longitude": 106.7091},
    "quan binh tan": {"Latitude": 10.7653, "Longitude": 106.6038},
    "quan go vap": {"Latitude": 10.8387, "Longitude": 106.6653},
    "quan phu nhuan": {"Latitude": 10.7992, "Longitude": 106.6803},
    "quan tan binh": {"Latitude": 10.8016, "Longitude": 106.6520},
    "quan tan phu": {"Latitude": 10.7901, "Longitude": 106.6282},
    "quan thu duc": {"Latitude": 10.8494, "Longitude": 106.7537},
    "huyen binh chanh": {"Latitude": 10.6956, "Longitude": 106.5740},
    "huyen can gio": {"Latitude": 10.4114, "Longitude": 106.9547},
    "huyen cu chi": {"Latitude": 10.9731, "Longitude": 106.4933},
    "huyen hoc mon": {"Latitude": 10.8831, "Longitude": 106.5864},
    "huyen nha be": {"Latitude": 10.6951, "Longitude": 106.7388},
    # TP.HCM Mới = TP Thủ Đức (gộp Q2, Q9, Thủ Đức cũ) — centroid toàn thành phố
    "tp ho chi minh moi": {"Latitude": 10.8494, "Longitude": 106.7537},
    "other": {"Latitude": HCMC_CENTER_LAT, "Longitude": HCMC_CENTER_LON},
}


def normalize_location(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"Huyen\s+Thanh\s+Quan", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Huyện\s+Thanh\s+Quan", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bQ(?:uan|uận)?[\.\s]*(\d{1,2})\b", r"Quan \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bQuan\s+(\d{1,2})\b", r"Quan \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bQuận\s+(\d{1,2})\b", r"Quan \1", text, flags=re.IGNORECASE)
    text = re.sub(
        r"TP[\.\s]*HCM|TP[\.\s]*Ho Chi Minh|TP[\.\s]*Hồ Chí Minh(\(Mới\))?",
        "TP Ho Chi Minh",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def remove_accents(value):
    text = normalize_location(value)
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def location_key(value):
    key = remove_accents(value).lower()
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


WARD_KEY_PATTERN = re.compile(
    r"\b(phuong\s+[a-z0-9\s]+)\b",
    flags=re.IGNORECASE,
)

def extract_district(location):
    key = location_key(location)
    
    # Bước 1: Tìm huyện/quận bằng cách match từng key trong VALID_DISTRICT_KEYS
    # Sắp xếp theo độ dài giảm dần để match key dài trước (ưu tiên huyện tên dài)
    sorted_keys = sorted(VALID_DISTRICT_KEYS.keys(), key=len, reverse=True)
    for dk in sorted_keys:
        if dk in key:
            return VALID_DISTRICT_KEYS[dk]
    
    # Bước 2: Fallback - Thử trích phường/xã để xác định vị trí
    if "tp ho chi minh" in key or "tphcm" in key:
        for match in WARD_KEY_PATTERN.finditer(key):
            ward_key = re.sub(r"\s+", " ", match.group(1)).strip()
            if ward_key in VALID_WARD_KEYS:
                return VALID_WARD_KEYS[ward_key]
        return "TP. Hồ Chí Minh (Mới)"
    
    return "Khác"


def district_lookup_key(value):
    return location_key(value)


def is_valid_hcm_coordinate(latitude, longitude):
    latitude = pd.to_numeric(latitude, errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")
    return latitude.between(*HCMC_LAT_RANGE) & longitude.between(*HCMC_LON_RANGE)


def haversine_distance_km(lat, lon, center_lat=HCMC_CENTER_LAT, center_lon=HCMC_CENTER_LON):
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = radians(center_lat)
    lon2 = radians(center_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def remove_impossible_rows(df, target_col=TARGET_COL):
    cleaned = df.copy()
    for col in [target_col, "Area"]:
        if col in cleaned.columns:
            cleaned = cleaned[pd.to_numeric(cleaned[col], errors="coerce") > 0]
    return cleaned


def replace_impossible_values(df):
    cleaned = df.copy()

    for col in ["Area", "Width", "Length", "Alley Width", "Agent Listing Count"]:
        if col in cleaned.columns:
            values = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = values.mask(values <= 0)

    for col in ["Bedrooms", "Bathrooms", "Floors"]:
        if col in cleaned.columns:
            values = pd.to_numeric(cleaned[col], errors="coerce")
            cleaned[col] = values.mask(values < 0)

    if {"Latitude", "Longitude"}.issubset(cleaned.columns):
        latitude = pd.to_numeric(cleaned["Latitude"], errors="coerce")
        longitude = pd.to_numeric(cleaned["Longitude"], errors="coerce")
        invalid_coords = ~is_valid_hcm_coordinate(latitude, longitude)
        cleaned["Latitude"] = latitude.mask(invalid_coords)
        cleaned["Longitude"] = longitude.mask(invalid_coords)

    return cleaned


def add_location_features(df):
    featured = df.copy()
    if "Location" in featured.columns:
        featured["District"] = featured["Location"].apply(extract_district)
    elif "District" not in featured.columns:
        featured["District"] = "Khác"
    return featured


def add_price_per_m2(df, target_col=TARGET_COL):
    featured = df.copy()
    if target_col in featured.columns and "Area" in featured.columns:
        price = pd.to_numeric(featured[target_col], errors="coerce")
        area = pd.to_numeric(featured["Area"], errors="coerce").replace(0, np.nan)
        featured["price_per_m2"] = price / area
    return featured


def is_land_property_type(series):
    return series.fillna("").map(location_key).eq("dat")


def fix_land_property_rooms(df):
    fixed = df.copy()
    if "Property Type" not in fixed.columns:
        return fixed

    land_mask = is_land_property_type(fixed["Property Type"])
    for col in ["Bedrooms", "Bathrooms", "Floors"]:
        if col in fixed.columns:
            fixed.loc[land_mask, col] = 0
    return fixed


def fit_quantile_bounds(series, lower_q=0.01, upper_q=0.99):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"lower": np.nan, "upper": np.nan}
    return {"lower": values.quantile(lower_q), "upper": values.quantile(upper_q)}


def clip_series(series, bounds):
    if pd.isna(bounds["lower"]) or pd.isna(bounds["upper"]):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce").clip(bounds["lower"], bounds["upper"])


def get_known_district_coordinates(df):
    district_coords = KNOWN_DISTRICT_COORDS.copy()
    # Bổ sung tọa độ phường/xã của TP Thủ Đức
    district_coords.update(KNOWN_WARD_COORDS)

    if not {"District", "Latitude", "Longitude"}.issubset(df.columns):
        return district_coords

    coords = df[["District", "Latitude", "Longitude"]].copy()
    coords["Latitude"] = pd.to_numeric(coords["Latitude"], errors="coerce")
    coords["Longitude"] = pd.to_numeric(coords["Longitude"], errors="coerce")
    coords = coords[is_valid_hcm_coordinate(coords["Latitude"], coords["Longitude"])]

    learned_coords = (
        coords.dropna(subset=["Latitude", "Longitude"])
        .groupby("District")[["Latitude", "Longitude"]]
        .median()
        .to_dict("index")
    )
    district_coords.update(
        {
            district_lookup_key(district): learned_coord
            for district, learned_coord in learned_coords.items()
        }
    )
    return district_coords


class BasicHousingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_col=TARGET_COL,
        lower_q=0.01,
        upper_q=0.99,
        hcm_center_lat=HCMC_CENTER_LAT,
        hcm_center_lon=HCMC_CENTER_LON,
    ):
        self.target_col = target_col
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.hcm_center_lat = hcm_center_lat
        self.hcm_center_lon = hcm_center_lon

    def fit(self, X, y=None):
        df = self._add_basic_features(X)

        self.numeric_medians_ = {}
        for col in BASE_NUMERIC_COLS:
            if col in df.columns and col not in ["Latitude", "Longitude", "price_per_m2"]:
                self.numeric_medians_[col] = pd.to_numeric(df[col], errors="coerce").median()

        self.district_coordinate_lookup_ = get_known_district_coordinates(df)

        self.global_latitude_ = (
            pd.to_numeric(df.get("Latitude"), errors="coerce").median()
            if "Latitude" in df.columns
            else np.nan
        )
        self.global_longitude_ = (
            pd.to_numeric(df.get("Longitude"), errors="coerce").median()
            if "Longitude" in df.columns
            else np.nan
        )
        if pd.isna(self.global_latitude_):
            self.global_latitude_ = self.hcm_center_lat
        if pd.isna(self.global_longitude_):
            self.global_longitude_ = self.hcm_center_lon

        if "price_per_m2" in df.columns:
            self.global_price_per_m2_bounds_ = fit_quantile_bounds(
                df["price_per_m2"],
                self.lower_q,
                self.upper_q,
            )
            self.district_price_per_m2_bounds_ = {
                district: fit_quantile_bounds(group["price_per_m2"], self.lower_q, self.upper_q)
                for district, group in df.groupby("District")
            }
        else:
            self.global_price_per_m2_bounds_ = {"lower": np.nan, "upper": np.nan}
            self.district_price_per_m2_bounds_ = {}

        numeric_cols = [
            col
            for col in OUTLIER_CLIP_COLS
            if col in df.columns and col not in [self.target_col, "price_per_m2"]
        ]
        self.numeric_bounds_ = {
            col: fit_quantile_bounds(df[col], self.lower_q, self.upper_q)
            for col in numeric_cols
        }

        self.target_bounds_ = (
            fit_quantile_bounds(df[self.target_col], self.lower_q, self.upper_q)
            if self.target_col in df.columns
            else {"lower": np.nan, "upper": np.nan}
        )
        return self

    def transform(self, X):
        df = self._add_basic_features(X)

        for col, bounds in self.numeric_bounds_.items():
            if col in df.columns:
                df[col] = clip_series(df[col], bounds)

        if self.target_col in df.columns:
            df[self.target_col] = clip_series(df[self.target_col], self.target_bounds_)

        df = self._fill_lat_lon_by_district(df)

        if "price_per_m2" in df.columns:
            df["price_per_m2"] = df.apply(self._clip_price_per_m2_row, axis=1)

        for col, value in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)

        # Apply after missing-value imputation so land rows are always exactly 0.
        df = fix_land_property_rooms(df)

        for col in BASE_CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        if {"Latitude", "Longitude"}.issubset(df.columns):
            df["distance_to_center"] = haversine_distance_km(
                df["Latitude"],
                df["Longitude"],
                self.hcm_center_lat,
                self.hcm_center_lon,
            )

        return df

    def _add_basic_features(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        df = replace_impossible_values(df)
        df = add_location_features(df)
        df = add_price_per_m2(df, self.target_col)
        return df

    def _fill_lat_lon_by_district(self, df):
        filled = df.copy()
        for col, global_value in [("Latitude", self.global_latitude_), ("Longitude", self.global_longitude_)]:
            if col not in filled.columns:
                filled[col] = global_value
            filled[col] = pd.to_numeric(filled[col], errors="coerce")

        missing_coords = filled[filled[["Latitude", "Longitude"]].isna().any(axis=1)]
        for idx, row in missing_coords.iterrows():
            district = district_lookup_key(row.get("District", "Khác"))
            district_coords = self.district_coordinate_lookup_.get(district, {})
            if pd.isna(row["Latitude"]):
                filled.at[idx, "Latitude"] = district_coords.get("Latitude", self.global_latitude_)
            if pd.isna(row["Longitude"]):
                filled.at[idx, "Longitude"] = district_coords.get("Longitude", self.global_longitude_)

        return filled

    def _clip_price_per_m2_row(self, row):
        bounds = self.district_price_per_m2_bounds_.get(
            row.get("District", "Khác"),
            self.global_price_per_m2_bounds_,
        )
        value = pd.to_numeric(row.get("price_per_m2"), errors="coerce")
        if pd.isna(value):
            return value
        return clip_series(pd.Series([value]), bounds).iloc[0]


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or DROP_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        return df.drop(columns=[col for col in self.columns if col in df.columns])


def get_feature_columns(df):
    numeric_features = [col for col in BASE_NUMERIC_COLS if col in df.columns]
    categorical_features = [col for col in BASE_CATEGORICAL_COLS if col in df.columns]
    return numeric_features, categorical_features


def build_preprocessing_pipeline(raw_df=None):
    if raw_df is not None:
        preview = BasicHousingPreprocessor().fit_transform(raw_df)
        preview = ColumnDropper().fit_transform(preview)
        numeric_features, categorical_features = get_feature_columns(preview)
    else:
        numeric_features = BASE_NUMERIC_COLS
        categorical_features = BASE_CATEGORICAL_COLS

    encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False)
        encoder_kwargs["sparse_output"] = False
    except TypeError:
        encoder_kwargs["sparse"] = False

    return Pipeline(
        steps=[
            ("basic", BasicHousingPreprocessor()),
            ("drop_columns", ColumnDropper()),
            (
                "encode_scale",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_features,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("one_hot", OneHotEncoder(**encoder_kwargs)),
                                ]
                            ),
                            categorical_features,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )


def prepare_model_data(df, target_col=TARGET_COL):
    cleaned = remove_impossible_rows(df, target_col=target_col).dropna(subset=[target_col])
    X = cleaned.drop(columns=[target_col])
    y = pd.to_numeric(cleaned[target_col], errors="coerce")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]
