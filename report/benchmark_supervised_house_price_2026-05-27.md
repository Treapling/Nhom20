# Benchmark va Chan doan du an du doan gia nha (2026-05-27)

## 1) Benchmark chi tiet: Du an hien tai vs du an tuong duong

### 1.1 Du an hien tai (noi bo)

Nguon metric:
- notebooks/Model_Comparison/results/model_comparison/cv_metrics.xlsx
- notebooks/Model_Comparison/results/model_comparison/error_analysis_summary.xlsx
- notebooks/Model_Comparison/Model_Comparison.ipynb (output cell Stacking)

#### CV (5-fold) - Mean CV MAE (don vi: trieu VND)

| Rank | Model | Mean CV MAE | Std CV MAE |
|---|---|---:|---:|
| 1 | Random Forest | 25,370.05 | 6,175.44 |
| 2 | XGBoost | 25,434.27 | 6,244.52 |
| 3 | LightGBM | 25,785.06 | 6,264.46 |
| 4 | Gradient Boosting | 26,829.74 | 6,301.77 |
| 5 | Linear Regression | 28,962.26 | 6,416.36 |
| 6 | Ridge | 28,962.84 | 6,415.86 |
| 7 | Lasso | 30,838.68 | 6,332.30 |
| 8 | Dummy Regressor | 30,838.68 | 6,332.30 |

#### Hold-out error summary (don vi: trieu VND)

| Rank by R2 | Model | RMSE | MAPE (%) | R2 |
|---|---|---:|---:|---:|
| 1 | Random Forest | 141,614.90 | 131.32 | 0.1533 |
| 2 | XGBoost | 147,166.51 | 124.40 | 0.0856 |
| 3 | LightGBM | 150,822.75 | 153.78 | 0.0396 |
| 4 | Gradient Boosting | 152,039.47 | 161.10 | 0.0241 |
| 5 | Ridge | 155,254.02 | 173.90 | -0.0176 |
| 6 | Linear Regression | 155,259.37 | 173.86 | -0.0177 |
| 7 | Lasso | 155,353.44 | 204.45 | -0.0189 |
| 8 | Dummy Regressor | 155,353.44 | 204.45 | -0.0189 |

#### Stacking (output da luu trong notebook)

- MAE: 20,222.61
- RMSE: 149,703.76
- MAPE: 147.16%
- R2: 0.0538

Nhan xet nhanh:
- MAE cua Stacking tot hon model don le, nhung RMSE/MAPE va R2 khong vuot Random Forest.
- CV MAE kha dep, nhung hold-out metric van yeu (R2 thap, MAPE rat cao).

---

### 1.2 So sanh voi cac du an tuong duong (public)

| Du an tham chieu | Mo hinh chinh | Metric cong bo | Ghi chu doi chieu |
|---|---|---|---|
| Utsav37/Zillow-House-Price-Prediction- | Ensemble LightGBM + RandomForest + XGBoost | README neu "accuracy 91%" | Chi la metric tu cong bo, chua ro dinh nghia "accuracy" voi bai toan regression |
| AshiteshSingh/House-price-prediction | Stacking: XGBoost, LightGBM, GBR, ExtraTrees, DNN | README: R2=0.9682, MAE~15.2 lakhs, MAPE~7.87% | Pipeline nang cao hon, metric tot hon rat nhieu (can luu y dataset khac) |
| neerajkesav/house-price-prediction | Ensemble XGBoost + LightGBM (Kaggle House Prices) | README: LB score = 0.12544 | Chuan so sanh theo leaderboard Kaggle, khong dong nhat metric voi du an hien tai |

Tong ket benchmark:
- Ve phuong phap: du an hien tai da di dung huong (tree ensemble + stacking + phan tich theo phan khuc/quan).
- Ve ket qua: du an hien tai thua benchmark public ve do on dinh hold-out (R2 va MAPE).

---

## 2) Vi sao CV dep nhung hold-out van yeu? (chan doan co so lieu)

### 2.1 Bang chung tu du lieu

Nguon: data/processed/train_v2.csv, data/processed/test_v2.csv (Price dang log1p)

- Phan bo phan khuc train/test kha giong nhau, nhung heavy-tail rat manh o nhom gia cao.
- Quy mo outlier cuc lon khong nho:
  - train_v2: >=200,000 trieu chiem 2.419%, >=1,000,000 trieu chiem 0.237%, max ~125,235,000 trieu
  - test_v2: >=200,000 trieu chiem 2.424%, >=1,000,000 trieu chiem 0.186%, max ~10,172,000 trieu
- MAPE rat nhay voi gia tri thuc te nho va outlier lon, dan den MAPE hold-out rat cao (124% - 204%).

### 2.2 Nguyen nhan ky thuat kha nang cao

1. Metric mismatch va sensitivity:
- CV toi uu tren MAE (muc tieu trung vi loi tuyet doi), nhung hold-out danh gia them RMSE/MAPE.
- RMSE phat nang outlier, MAPE phat nang mau co y nho -> metric hold-out xau la de hieu.

2. Du lieu heavy-tail va outlier cuc doan:
- Outlier co bien do cuc lon khien RMSE va R2 bi "keo" manh, du MAE van on.

3. Clipping trong khong gian log:
- Dang dung clip y_hat_log vao [0, 25]. Nguong tren 25 sau expm1 van rat lon, nen khong that su khu outlier prediction theo don vi goc.

4. Stacking toi uu chua dung muc tieu hold-out:
- Stacking cai thien MAE nhung khong cai thien RMSE/R2 so voi Random Forest, cho thay can toi uu muc tieu va regularization theo metric business.

---

## 3) De xuat cai thien uu tien (ngan han)

1. Doi bo metric toi uu:
- Chon metric chinh theo muc tieu kinh doanh (de xuat: MAE + WMAPE), khong dung MAPE thuan lam metric chinh.
- Bao cao song song: MAE, RMSE, R2, WMAPE, MedianAE.

2. Kiem soat outlier muc tieu:
- Loc/winsorize nguong tren cua Price theo percentile (vi du p99.5/p99.9) tren train.
- Danh gia them model robust (Huber, Quantile loss, Tweedie/Poisson neu phu hop).

3. Stratified CV theo phan khuc gia:
- Thay KFold bang stratified bins theo target (gia) de giam sai lech giua fold.
- Bao cao do lech metric theo tung fold va tung phan khuc.

4. Danh gia theo segment la bat buoc:
- Tinh MAE/WMAPE theo 5 phan khuc gia va theo quan/huyen cho moi model.
- Chon model production dua tren tong hop (khong chi diem trung binh toan tap).

5. Hieu chinh stacking:
- Tune lai base learners va meta-learner theo muc tieu MAE/WMAPE.
- Dung OOF predictions dung chuan va regularization manh hon cho meta model.

---

## 4) Ket luan

- Du an hien tai da co nen tang phuong phap tot va day du.
- Van de lon nhat khong nam o thieu mo hinh, ma nam o bo metric + heavy-tail outlier + cach danh gia.
- Neu thuc hien 5 buoc uu tien tren, kha nang cao se nang duoc hold-out R2 va giam MAPE ro ret ma khong can thay doi kien truc qua lon.
