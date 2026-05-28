import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Thiết lập style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Dữ liệu từ báo cáo phân tích
missing_data = {
    'Length': 26037,
    'Direction': 40630,
    'Bedrooms': 34662,
    'Bathrooms': 40540,
    'Floors': 42624,
    'Latitude/Longitude': 29856,
    'Area': 104,
    'Agent Name': 929
}

outlier_data = {
    'Price (Max)': 660000000,
    'Area (Max)': 1578861000,
    'Length (Max)': 90000,
    'Width (Max)': 50000,
    'Bedrooms (Max)': 6902,
    'Alley Width (Max)': 999999.99
}

model_performance = {
    'MAE': 13145.64,
    'RMSE': 45214.97,
    'MAPE': 356.80,
    'R² Score': -0.0052
}

segment_mae = {
    'Thấp (<5B)': 3137.91,
    'Trung Bình (5-15B)': 4172.64,
    'Cao (>15B)': 44932.27
}

# Tạo figure với 4 subplots
fig = plt.figure()
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# 1. Missing Data
ax1 = fig.add_subplot(gs[0, :])
missing_sorted = dict(sorted(missing_data.items(), key=lambda x: x[1], reverse=True))
colors1 = ['#d62728' if v > 30000 else '#ff7f0e' if v > 10000 else '#2ca02c' for v in missing_sorted.values()]
ax1.barh(list(missing_sorted.keys()), list(missing_sorted.values()), color=colors1)
ax1.set_xlabel('Số lượng giá trị thiếu', fontsize=11, fontweight='bold')
ax1.set_title('1. Dữ Liệu Thiếu (Missing Data) - Vấn Đề Nghiêm Trọng Nhất', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(missing_sorted.values()):
    percentage = (v / 51304) * 100
    ax1.text(v, i, f'  {v:,} ({percentage:.1f}%)', va='center', fontsize=9)

# 2. Outliers
ax2 = fig.add_subplot(gs[1, 0])
outlier_keys = list(outlier_data.keys())
outlier_vals = list(outlier_data.values())
colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(outlier_keys)))
ax2.barh(outlier_keys, outlier_vals, color=colors2)
ax2.set_xlabel('Giá Trị Tối Đa', fontsize=10, fontweight='bold')
ax2.set_title('2. Dữ Liệu Ngoại Lệ (Outliers)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Model Performance
ax3 = fig.add_subplot(gs[1, 1])
metrics = list(model_performance.keys())
values = list(model_performance.values())
# Normalize values for display
display_values = [13145.64/10000, 45214.97/10000, 356.80/100, -0.0052]
colors3 = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
bars = ax3.bar(metrics, display_values, color=colors3, alpha=0.7)
ax3.set_ylabel('Giá Trị (Chuẩn Hóa)', fontsize=10, fontweight='bold')
ax3.set_title('3. Hiệu Năng Mô Hình Linear Regression', fontsize=12, fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. MAE by Price Segment
ax4 = fig.add_subplot(gs[2, 0])
segments = list(segment_mae.keys())
mae_values = list(segment_mae.values())
colors4 = ['#2ca02c', '#ff7f0e', '#d62728']
bars4 = ax4.bar(segments, mae_values, color=colors4, alpha=0.7)
ax4.set_ylabel('MAE (Triệu VND)', fontsize=10, fontweight='bold')
ax4.set_title('4. Lỗi Dự Báo (MAE) Theo Phân Khúc Giá', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars4, mae_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f'{val:,.0f}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 5. Severity Summary
ax5 = fig.add_subplot(gs[2, 1])
severity = {
    'Rất Cao': 11,
    'Cao': 9,
    'Trung Bình': 7,
    'Thấp': 3
}
colors5 = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
wedges, texts, autotexts = ax5.pie(severity.values(), labels=severity.keys(), autopct='%1.1f%%',
                                     colors=colors5, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
ax5.set_title('5. Phân Bố Mức Độ Nghiêm Trọng', fontsize=12, fontweight='bold')

# Add overall title
fig.suptitle('PHÂN TÍCH LỖI VÀ MẪU SAI - DỰ ÁN DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('error_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("Biểu đồ đã được lưu thành công: error_analysis_visualization.png")

# Thêm thống kê tóm tắt
print("\n" + "="*70)
print("TÓM TẮT PHÂN TÍCH LỖI VÀ MẪU SAI")
print("="*70)
print(f"\n📊 TỔNG SỐ LỖI PHÁT HIỆN: 30 lỗi")
print(f"   - Mức độ Rất Cao: 11 lỗi (36.7%)")
print(f"   - Mức độ Cao: 9 lỗi (30%)")
print(f"   - Mức độ Trung Bình: 7 lỗi (23.3%)")
print(f"   - Mức độ Thấp: 3 lỗi (10%)")

print(f"\n❌ VẤN ĐỀ CHỦ YẾU:")
print(f"   1. Dữ liệu thiếu cao (42,624 hàng mất Floors - 83.1%)")
print(f"   2. Outlier quá lớn (Area max: 1.5 tỷ m², không thực tế)")
print(f"   3. Mô hình Linear Regression không phù hợp (R² = -0.0052)")
print(f"   4. Dự báo kém cho phân khúc cao (MAE = 44,932)")
print(f"   5. MAPE = 356.8% (rất tệ)")

print(f"\n✅ KHUYẾN NGHỊ CẤP BÁCH:")
print(f"   1. Xử lý thiếu dữ liệu: Loại bỏ hoặc imputation")
print(f"   2. Làm sạch outlier: Sử dụng IQR hoặc Z-score")
print(f"   3. Thay đổi mô hình: Random Forest, XGBoost")
print(f"   4. Feature engineering: Tạo đặc trưng mới")
print(f"   5. Xử lý riêng phân khúc: Model riêng cho từng segment")
print("="*70 + "\n")
