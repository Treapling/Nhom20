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
    "Agent Role",
    "Ward",   # high-cardinality helper; used only for coord fill, then dropped
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

# ── Phường thuộc TP Thủ Đức (TP.HCM Mới) ──────────────────────────────────────
# Dùng cho extract_district() fallback khi không match được quận cũ
VALID_WARD_KEYS = {
    # Quận 9 cũ
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
    # Quận 2 cũ
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
    # Thủ Đức cũ
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

# ── Phường/xã toàn bộ TP.HCM (dùng cho extract_ward) ──────────────────────────
# Quận 1
_Q1 = {
    "phuong ben nghe": "Phường Bến Nghé",
    "phuong ben thanh": "Phường Bến Thành",
    "phuong co giang": "Phường Cô Giang",
    "phuong cau kho": "Phường Cầu Kho",
    "phuong cau ong lanh": "Phường Cầu Ông Lãnh",
    "phuong da kao": "Phường Đa Kao",
    "phuong nguyen cu trinh": "Phường Nguyễn Cư Trinh",
    "phuong nguyen thai binh": "Phường Nguyễn Thái Bình",
    "phuong pham ngu lao": "Phường Phạm Ngũ Lão",
    "phuong tan dinh": "Phường Tân Định",
}
# Quận 3
_Q3 = {
    "phuong 1": "Phường 1",
    "phuong 2": "Phường 2",
    "phuong 3": "Phường 3",
    "phuong 4": "Phường 4",
    "phuong 5": "Phường 5",
    "phuong 6": "Phường 6",
    "phuong 7": "Phường 7",
    "phuong 8": "Phường 8",
    "phuong 9": "Phường 9",
    "phuong 10": "Phường 10",
    "phuong 11": "Phường 11",
    "phuong 12": "Phường 12",
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong vo thi sau": "Phường Võ Thị Sáu",
}
# Quận 4
_Q4 = {
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong 15": "Phường 15",
    "phuong 16": "Phường 16",
    "phuong 18": "Phường 18",
}
# Quận 7
_Q7 = {
    "phuong binh thuan": "Phường Bình Thuận",
    "phuong phu my": "Phường Phú Mỹ",
    "phuong phu thuan": "Phường Phú Thuận",
    "phuong tan hung": "Phường Tấn Hưng",
    "phuong tan kieng": "Phường Tân Kiểng",
    "phuong tan phu": "Phường Tân Phú",
    "phuong tan quy": "Phường Tân Quy",
    "phuong tan thuan dong": "Phường Tân Thuận Đông",
    "phuong tan thuan tay": "Phường Tân Thuận Tây",
}
# Quận Bình Thạnh
_QBT = {
    "phuong 1": "Phường 1",
    "phuong 2": "Phường 2",
    "phuong 3": "Phường 3",
    "phuong 5": "Phường 5",
    "phuong 6": "Phường 6",
    "phuong 7": "Phường 7",
    "phuong 11": "Phường 11",
    "phuong 12": "Phường 12",
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong 15": "Phường 15",
    "phuong 17": "Phường 17",
    "phuong 19": "Phường 19",
    "phuong 21": "Phường 21",
    "phuong 22": "Phường 22",
    "phuong 24": "Phường 24",
    "phuong 25": "Phường 25",
    "phuong 26": "Phường 26",
    "phuong 27": "Phường 27",
    "phuong 28": "Phường 28",
}
# Quận Gò Vấp
_QGV = {
    "phuong 1": "Phường 1",
    "phuong 3": "Phường 3",
    "phuong 4": "Phường 4",
    "phuong 5": "Phường 5",
    "phuong 6": "Phường 6",
    "phuong 7": "Phường 7",
    "phuong 8": "Phường 8",
    "phuong 9": "Phường 9",
    "phuong 10": "Phường 10",
    "phuong 11": "Phường 11",
    "phuong 12": "Phường 12",
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong 15": "Phường 15",
    "phuong 16": "Phường 16",
    "phuong 17": "Phường 17",
}
# Quận Phú Nhuận
_QPN = {
    "phuong 1": "Phường 1",
    "phuong 2": "Phường 2",
    "phuong 3": "Phường 3",
    "phuong 4": "Phường 4",
    "phuong 5": "Phường 5",
    "phuong 7": "Phường 7",
    "phuong 8": "Phường 8",
    "phuong 9": "Phường 9",
    "phuong 10": "Phường 10",
    "phuong 11": "Phường 11",
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong 15": "Phường 15",
    "phuong 17": "Phường 17",
}
# Quận Tân Bình
_QTB = {
    "phuong 1": "Phường 1",
    "phuong 2": "Phường 2",
    "phuong 3": "Phường 3",
    "phuong 4": "Phường 4",
    "phuong 5": "Phường 5",
    "phuong 6": "Phường 6",
    "phuong 7": "Phường 7",
    "phuong 8": "Phường 8",
    "phuong 9": "Phường 9",
    "phuong 10": "Phường 10",
    "phuong 11": "Phường 11",
    "phuong 12": "Phường 12",
    "phuong 13": "Phường 13",
    "phuong 14": "Phường 14",
    "phuong 15": "Phường 15",
}
# Quận Tân Phú
_QTP = {
    "phuong hiep tan": "Phường Hiệp Tân",
    "phuong hoa thanh": "Phường Hòa Thạnh",
    "phuong phu tho hoa": "Phường Phú Thọ Hòa",
    "phuong phu trung": "Phường Phú Trung",
    "phuong son ky": "Phường Sơn Kỳ",
    "phuong tan quy": "Phường Tân Quý",
    "phuong tan son nhi": "Phường Tân Sơn Nhì",
    "phuong tan thanh dong": "Phường Tân Thành Đông",
    "phuong tan thanh tay": "Phường Tân Thành Tây",
    "phuong tay thanh": "Phường Tây Thạnh",
}
# Quận Bình Tân
_QBTan = {
    "phuong an lac": "Phường An Lạc",
    "phuong an lac a": "Phường An Lạc A",
    "phuong binh hung hoa": "Phường Bình Hưng Hòa",
    "phuong binh hung hoa a": "Phường Bình Hưng Hòa A",
    "phuong binh hung hoa b": "Phường Bình Hưng Hòa B",
    "phuong binh tri dong": "Phường Bình Trị Đông",
    "phuong binh tri dong a": "Phường Bình Trị Đông A",
    "phuong binh tri dong b": "Phường Bình Trị Đông B",
    "phuong tan tao": "Phường Tân Tạo",
    "phuong tan tao a": "Phường Tân Tạo A",
}
# Huyện Bình Chánh (xã)
_HBC = {
    "xa an phu tay": "Xã An Phú Tây",
    "xa binh chanh": "Xã Bình Chánh",
    "xa binh hung": "Xã Bình Hưng",
    "xa binh loi": "Xã Bình Lợi",
    "xa da phuoc": "Xã Đa Phước",
    "xa hung long": "Xã Hưng Long",
    "xa le minh xuan": "Xã Lê Minh Xuân",
    "xa phong phu": "Xã Phong Phú",
    "xa qui duc": "Xã Quy Đức",
    "xa tan kien": "Xã Tân Kiên",
    "xa tan nhat": "Xã Tân Nhật",
    "xa tan quy tay": "Xã Tân Quý Tây",
    "xa vinh loc a": "Xã Vĩnh Lộc A",
    "xa vinh loc b": "Xã Vĩnh Lộc B",
    "thi tran tan tuc": "Thị Trấn Tân Túc",
}
# Huyện Nhà Bè (xã)
_HNB = {
    "xa hiep phuoc": "Xã Hiệp Phước",
    "xa long thoi": "Xã Long Thới",
    "xa nhon duc": "Xã Nhơn Đức",
    "xa phu xuan": "Xã Phú Xuân",
    "xa phuoc kien": "Xã Phước Kiến",
    "xa phuoc loc": "Xã Phước Lộc",
    "thi tran nha be": "Thị Trấn Nhà Bè",
}

# Tổng hợp toàn bộ phường/xã HCM (dùng cho extract_ward)
VALID_WARD_KEYS_ALL_HCM = {}
VALID_WARD_KEYS_ALL_HCM.update(VALID_WARD_KEYS)   # TP Thủ Đức (Q2+Q9+TD cũ)
VALID_WARD_KEYS_ALL_HCM.update(_Q1)
VALID_WARD_KEYS_ALL_HCM.update(_Q3)
VALID_WARD_KEYS_ALL_HCM.update(_Q4)
VALID_WARD_KEYS_ALL_HCM.update(_Q7)
VALID_WARD_KEYS_ALL_HCM.update(_QBT)
VALID_WARD_KEYS_ALL_HCM.update(_QGV)
VALID_WARD_KEYS_ALL_HCM.update(_QPN)
VALID_WARD_KEYS_ALL_HCM.update(_QTB)
VALID_WARD_KEYS_ALL_HCM.update(_QTP)
VALID_WARD_KEYS_ALL_HCM.update(_QBTan)
VALID_WARD_KEYS_ALL_HCM.update(_HBC)
VALID_WARD_KEYS_ALL_HCM.update(_HNB)

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
    # Quận 1
    "phuong ben nghe":         {"Latitude": 10.7719, "Longitude": 106.7033},
    "phuong ben thanh":        {"Latitude": 10.7726, "Longitude": 106.6983},
    "phuong co giang":         {"Latitude": 10.7640, "Longitude": 106.6945},
    "phuong cau kho":          {"Latitude": 10.7641, "Longitude": 106.6993},
    "phuong da kao":           {"Latitude": 10.7885, "Longitude": 106.7005},
    "phuong nguyen cu trinh":  {"Latitude": 10.7627, "Longitude": 106.6937},
    "phuong nguyen thai binh": {"Latitude": 10.7722, "Longitude": 106.6969},
    "phuong pham ngu lao":     {"Latitude": 10.7686, "Longitude": 106.6932},
    "phuong tan dinh":         {"Latitude": 10.7897, "Longitude": 106.6937},
    # Quận 7
    "phuong binh thuan":       {"Latitude": 10.7271, "Longitude": 106.7215},
    "phuong phu my":           {"Latitude": 10.7202, "Longitude": 106.7277},
    "phuong phu thuan":        {"Latitude": 10.7327, "Longitude": 106.7133},
    "phuong tan hung":         {"Latitude": 10.7392, "Longitude": 106.7186},
    "phuong tan kieng":        {"Latitude": 10.7276, "Longitude": 106.7127},
    "phuong tan quy":          {"Latitude": 10.7382, "Longitude": 106.7099},
    "phuong tan thuan dong":   {"Latitude": 10.7479, "Longitude": 106.7155},
    "phuong tan thuan tay":    {"Latitude": 10.7411, "Longitude": 106.7145},
    # Quận Phú Nhuận
    "phuong 9":                {"Latitude": 10.7985, "Longitude": 106.6848},
    "phuong 10":               {"Latitude": 10.7935, "Longitude": 106.6839},
    "phuong 17":               {"Latitude": 10.8082, "Longitude": 106.6784},
    # Quận Tân Phú
    "phuong hiep tan":         {"Latitude": 10.7916, "Longitude": 106.6278},
    "phuong hoa thanh":        {"Latitude": 10.7878, "Longitude": 106.6263},
    "phuong phu tho hoa":      {"Latitude": 10.7944, "Longitude": 106.6247},
    "phuong phu trung":        {"Latitude": 10.7857, "Longitude": 106.6234},
    "phuong son ky":           {"Latitude": 10.8003, "Longitude": 106.6271},
    "phuong tan quy":          {"Latitude": 10.7827, "Longitude": 106.6288},
    "phuong tan son nhi":      {"Latitude": 10.7980, "Longitude": 106.6300},
    "phuong tay thanh":        {"Latitude": 10.8041, "Longitude": 106.6293},
    # Quận Bình Tân
    "phuong an lac":           {"Latitude": 10.7527, "Longitude": 106.6135},
    "phuong an lac a":         {"Latitude": 10.7491, "Longitude": 106.6063},
    "phuong binh hung hoa":    {"Latitude": 10.7800, "Longitude": 106.6035},
    "phuong binh hung hoa a":  {"Latitude": 10.7755, "Longitude": 106.5978},
    "phuong binh hung hoa b":  {"Latitude": 10.7701, "Longitude": 106.6006},
    "phuong binh tri dong":    {"Latitude": 10.7654, "Longitude": 106.6079},
    "phuong binh tri dong a":  {"Latitude": 10.7608, "Longitude": 106.6027},
    "phuong binh tri dong b":  {"Latitude": 10.7566, "Longitude": 106.6053},
    "phuong tan tao":          {"Latitude": 10.7431, "Longitude": 106.6085},
    "phuong tan tao a":        {"Latitude": 10.7388, "Longitude": 106.6050},
    # Huyện Nhà Bè
    "xa hiep phuoc":           {"Latitude": 10.6631, "Longitude": 106.7420},
    "xa long thoi":            {"Latitude": 10.6827, "Longitude": 106.7350},
    "xa nhon duc":             {"Latitude": 10.6750, "Longitude": 106.7283},
    "xa phu xuan":             {"Latitude": 10.7012, "Longitude": 106.7413},
    "xa phuoc kien":           {"Latitude": 10.6943, "Longitude": 106.7453},
    "xa phuoc loc":            {"Latitude": 10.6809, "Longitude": 106.7457},
    "thi tran nha be":         {"Latitude": 10.7081, "Longitude": 106.7369},
    # Huyện Bình Chánh
    "xa an phu tay":           {"Latitude": 10.6652, "Longitude": 106.6153},
    "xa binh chanh":           {"Latitude": 10.6571, "Longitude": 106.5834},
    "xa binh hung":            {"Latitude": 10.7080, "Longitude": 106.6490},
    "xa binh loi":             {"Latitude": 10.6448, "Longitude": 106.5648},
    "xa da phuoc":             {"Latitude": 10.6477, "Longitude": 106.6439},
    "xa hung long":            {"Latitude": 10.6333, "Longitude": 106.5939},
    "xa le minh xuan":         {"Latitude": 10.6855, "Longitude": 106.5287},
    "xa phong phu":            {"Latitude": 10.7208, "Longitude": 106.6283},
    "xa tan kien":             {"Latitude": 10.7056, "Longitude": 106.5919},
    "xa tan nhat":             {"Latitude": 10.6601, "Longitude": 106.5607},
    "xa tan quy tay":          {"Latitude": 10.6757, "Longitude": 106.5780},
    "xa vinh loc a":           {"Latitude": 10.7434, "Longitude": 106.5724},
    "xa vinh loc b":           {"Latitude": 10.7338, "Longitude": 106.5679},
    "thi tran tan tuc":        {"Latitude": 10.6946, "Longitude": 106.5977},
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
    # Đ/đ has no NFKD decomposition to ASCII — replace manually before normalization
    text = text.replace("Đ", "D").replace("đ", "d")
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

# Pattern khớp phường, xã, thị trấn
WARD_ADMIN_PATTERN = re.compile(
    r"\b((phuong|xa|thi\s+tran)\s+[a-z0-9\s]+?)\b",
    flags=re.IGNORECASE,
)

# Sắp xếp VALID_WARD_KEYS_ALL_HCM keys một lần (dài → ngắn) để dùng khi match
_WARD_KEYS_SORTED = sorted(VALID_WARD_KEYS_ALL_HCM.keys(), key=len, reverse=True)


def extract_ward(location) -> str | None:
    """Trích tên phường/xã/thị trấn từ chuỗi địa chỉ HCM.

    Trả về tên chuẩn (có dấu, ví dụ "Phường Thảo Điền") nếu match được,
    hoặc None nếu không tìm thấy / không phải địa chỉ HCM.
    """
    if pd.isna(location):
        return None
    key = location_key(str(location))
    # Chỉ xử lý địa chỉ TP HCM
    if "tp ho chi minh" not in key and "tphcm" not in key:
        return None
    for wk in _WARD_KEYS_SORTED:
        if wk in key:
            return VALID_WARD_KEYS_ALL_HCM[wk]
    return None


def extract_district(location):
    key = location_key(location)

    # Bước 1: Tìm huyện/quận — match key dài trước
    sorted_keys = sorted(VALID_DISTRICT_KEYS.keys(), key=len, reverse=True)
    for dk in sorted_keys:
        if dk in key:
            return VALID_DISTRICT_KEYS[dk]

    # Bước 2: Fallback — thử trích phường TP Thủ Đức
    if "tp ho chi minh" in key or "tphcm" in key:
        for wk in sorted(VALID_WARD_KEYS.keys(), key=len, reverse=True):
            if wk in key:
                return VALID_WARD_KEYS[wk]
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

    if "Price" in cleaned.columns:
        price_vals = pd.to_numeric(cleaned["Price"], errors="coerce")
        # Drop rows where price is less than 100 (assumed unit: million or as dataset uses)
        cleaned = cleaned[price_vals >= 100]

    return cleaned


def add_location_features(df):
    """Thêm cột District và Ward từ cột Location.

    Ward: tên phường/xã/thị trấn chính xác, dùng để tra toạ độ mịn hơn
    quận/huyện.  Chỉ điền cho địa chỉ TP.HCM; các trường hợp khác để None.
    """
    featured = df.copy()
    if "Location" in featured.columns:
        featured["District"] = featured["Location"].apply(extract_district)
        featured["Ward"] = featured["Location"].apply(extract_ward)
    else:
        if "District" not in featured.columns:
            featured["District"] = "Khác"
        if "Ward" not in featured.columns:
            featured["Ward"] = None
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

def fit_iqr_bounds(series, iqr_k=1.5):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"lower": np.nan, "upper": np.nan}
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr
    return {"lower": lower, "upper": upper}

def clip_series(series, bounds):
    if pd.isna(bounds["lower"]) or pd.isna(bounds["upper"]):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series, errors="coerce").clip(bounds["lower"], bounds["upper"])


def get_known_district_coordinates(df):
    # Start with all static coords (district + all known wards)
    district_coords = KNOWN_DISTRICT_COORDS.copy()
    district_coords.update(KNOWN_WARD_COORDS)   # ward keys override district keys

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
        # Also learn ward-level median coords from train data where available
        if {"Ward", "Latitude", "Longitude"}.issubset(df.columns):
            ward_coords_raw = df[["Ward", "Latitude", "Longitude"]].copy()
            ward_coords_raw = ward_coords_raw[ward_coords_raw["Ward"].notna()]
            ward_coords_raw["Latitude"] = pd.to_numeric(ward_coords_raw["Latitude"], errors="coerce")
            ward_coords_raw["Longitude"] = pd.to_numeric(ward_coords_raw["Longitude"], errors="coerce")
            ward_coords_raw = ward_coords_raw[
                is_valid_hcm_coordinate(ward_coords_raw["Latitude"], ward_coords_raw["Longitude"])
            ]
            learned_wards = (
                ward_coords_raw.dropna(subset=["Latitude", "Longitude"])
                .groupby("Ward")[["Latitude", "Longitude"]]
                .median()
                .to_dict("index")
            )
            # Store as ward_key → coords for use in _fill_lat_lon_by_district
            self.ward_coordinate_lookup_ = {
                location_key(w): coords for w, coords in learned_wards.items()
            }
        else:
            self.ward_coordinate_lookup_ = {}

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
                lower_q=0.01,
                upper_q=0.99
            )
            self.district_price_per_m2_bounds_ = {
                district: fit_quantile_bounds(group["price_per_m2"], lower_q=0.01, upper_q=0.99)
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
            col: fit_quantile_bounds(df[col], lower_q=0.01, upper_q=0.99)
            for col in numeric_cols
        }

        self.target_bounds_ = (
            # fit_quantile_bounds(df[self.target_col], lower_q=0.01, upper_q=0.9)
            {"lower": 100, "upper": 40000} # Cố định bounds cho Price để tránh bị ảnh hưởng bởi outliers trong train
            if self.target_col in df.columns
            else {"lower": np.nan, "upper": np.nan}
        )
        return self

    def transform(self, X):
        df = self._add_basic_features(X)

        for col, bounds in self.numeric_bounds_.items():
            if col in df.columns:
                df[col] = clip_series(df[col], bounds)

        # if self.target_col in df.columns:
        #     df[self.target_col] = clip_series(df[self.target_col], self.target_bounds_)

        df = self._fill_lat_lon_by_district(df)

        # Remove outliers for price and price_per_m2 (instead of clipping)
        df = self._remove_price_outliers(df)

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

        # Drop Ward after coord fill — high-cardinality helper column
        df = df.drop(columns=["Ward"], errors="ignore")

        return df

    def _add_basic_features(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        df = replace_impossible_values(df)
        df = add_location_features(df)
        df = add_price_per_m2(df, self.target_col)
        return df

    def _fill_lat_lon_by_district(self, df):
        """Fill missing Latitude/Longitude theo thứ tự ưu tiên:
        1. Toạ độ phường/xã (ward) — chính xác nhất (~500 m)
        2. Toạ độ quận/huyện (district) — trung bình (~3–5 km)
        3. Global median train — fallback cuối cùng
        """
        filled = df.copy()
        for col, global_value in [("Latitude", self.global_latitude_), ("Longitude", self.global_longitude_)]:
            if col not in filled.columns:
                filled[col] = global_value
            filled[col] = pd.to_numeric(filled[col], errors="coerce")

        missing_mask = filled[["Latitude", "Longitude"]].isna().any(axis=1)
        if not missing_mask.any():
            return filled

        for idx, row in filled[missing_mask].iterrows():
            lat_miss = pd.isna(row["Latitude"])
            lon_miss = pd.isna(row["Longitude"])
            if not (lat_miss or lon_miss):
                continue

            # Priority 1: ward-level coords (learned first, then static)
            ward_raw = row.get("Ward")
            if ward_raw and not pd.isna(ward_raw):
                ward_key = location_key(str(ward_raw))
                ward_coords = (
                    getattr(self, "ward_coordinate_lookup_", {}).get(ward_key)
                    or KNOWN_WARD_COORDS.get(ward_key, {})
                )
                if ward_coords:
                    if lat_miss:
                        filled.at[idx, "Latitude"] = ward_coords["Latitude"]
                        lat_miss = False
                    if lon_miss:
                        filled.at[idx, "Longitude"] = ward_coords["Longitude"]
                        lon_miss = False

            if not (lat_miss or lon_miss):
                continue

            # Priority 2: district/learned coords
            district = district_lookup_key(row.get("District", "Khác"))
            district_coords = self.district_coordinate_lookup_.get(district, {})
            if lat_miss:
                filled.at[idx, "Latitude"] = district_coords.get("Latitude", self.global_latitude_)
            if lon_miss:
                filled.at[idx, "Longitude"] = district_coords.get("Longitude", self.global_longitude_)

        return filled

    def _remove_price_outliers(self, df):
        """Remove rows where Price or price_per_m2 fall outside bounds.

        Applied per-district if district bounds exist, otherwise uses global bounds.
        """
        filtered = df.copy()

        # 1. Xử lý target_col (Global Bounds) - Giữ nguyên logic cũ nhưng viết gọn lại
        if self.target_col in filtered.columns:
            target_vals = pd.to_numeric(filtered[self.target_col], errors="coerce")
            bounds = self.target_bounds_

            if not (pd.isna(bounds["lower"]) or pd.isna(bounds["upper"])):
                # Tạo mask: Giữ lại nếu nằm trong bound HOẶC là NaN
                mask = (
                    (target_vals >= bounds["lower"]) & (target_vals <= bounds["upper"])
                ) | target_vals.isna()
                filtered = filtered[mask]

        # 2. Xử lý price_per_m2 (Per-district hoặc Global Bounds) - TỐI ƯU HÓA Ở ĐÂY
        if "price_per_m2" in filtered.columns:
            # Ép kiểu toàn bộ cột một lần duy nhất (Nhanh hơn ép kiểu từng dòng)
            ppm_vals = pd.to_numeric(filtered["price_per_m2"], errors="coerce")

            # Tạo series chứa bounds cho từng dòng dựa vào cột 'District'
            # Nếu không tìm thấy quận, mặc định map về self.global_price_per_m2_bounds_
            district_series = filtered["District"].fillna("Khác")

            # Trích xuất nhanh lower và upper bound cho từng dòng dựa trên quận
            lower_bounds = district_series.map(
                lambda d: self.district_price_per_m2_bounds_.get(
                    d, self.global_price_per_m2_bounds_
                )["lower"]
            )
            upper_bounds = district_series.map(
                lambda d: self.district_price_per_m2_bounds_.get(
                    d, self.global_price_per_m2_bounds_
                )["upper"]
            )

            # Tạo điều kiện lọc (Vectorized)
            # Giữ lại nếu: (Giá trị trong khoảng bound) HOẶC (Giá trị là NaN) HOẶC (Bound bị NaN)
            is_within_bounds = (ppm_vals >= lower_bounds) & (ppm_vals <= upper_bounds)
            is_nan_val = ppm_vals.isna()
            is_nan_bound = lower_bounds.isna() | upper_bounds.isna()

            final_mask = is_within_bounds | is_nan_val | is_nan_bound

            # Lọc dữ liệu
            filtered = filtered[final_mask]

        return filtered.reset_index(drop=True)

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
