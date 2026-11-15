# Nhom01_GDG

Nhóm 1 nhận làm chủ đề **A1. Phát hiện gian lận: Imbalanced dataset**

## Đề bài

### A1. Phát hiện gian lận: Imbalanced dataset

| Hạng mục | Nội dung |
| --- | --- |
| **Câu hỏi** | Khi giao dịch gian lận chỉ chiếm <1% dữ liệu, làm sao để model không "lười" dự đoán tất cả là bình thường? |
| **Gợi ý dữ liệu** | Credit Card Fraud Detection (Kaggle), PaySim, hoặc IEEE-CIS Fraud Detection |
| **Gợi ý hướng thử nghiệm** | - Baseline: Logistic Regression trên dữ liệu gốc<br>- Resampling: undersampling, oversampling, SMOTE, SMOTE-Tomek<br>- Class weight / cost-sensitive learning (`scale_pos_weight`)<br>- Các model khác nhau (Random Forest, XGBoost, LightGBM, CatBoost)<br>- Điều chỉnh ngưỡng quyết định (cost-sensitive threshold tuning)<br>- Multi-model soft voting ensemble |
| **Thang đo** | Precision, Recall, F1, PR-AUC (giải thích vì sao Accuracy không dùng được); Confusion matrix |
| **Bắt buộc** | Chứng minh bằng số liệu vì sao Accuracy gây hiểu lầm; ít nhất 3 cách xử lý mất cân bằng; giải thích đánh đổi Precision và Recall dưới góc nhìn kinh doanh (bỏ sót gian lận và chặn nhầm khách hàng, cái nào tốn kém hơn?) |
| **Mở rộng** | Ước tính thiệt hại/tiết kiệm bằng tiền theo từng ngưỡng; giải thích model bằng SHAP |
| **Tham khảo** | Tài liệu thư viện imbalanced-learn; Chawla et al. (2002), SMOTE |

---

## 🛠 Hướng dẫn cài đặt môi trường & Sửa các lỗi thường gặp

### 1. Khởi tạo môi trường ảo (Khuyến nghị)
Để tránh xung đột package hệ thống, nên tạo virtual environment riêng:
```bash
python -m venv .venv
```
Kích hoạt môi trường:
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.\.venv\Scripts\activate.bat`
- **Linux / macOS:** `source .venv/bin/activate`

---

### 2. Cài đặt các thư viện (Packages)

#### Cách 1: Sử dụng pip (Khuyến nghị)
```bash
pip install -r requirements.txt
```

#### Các lỗi thường gặp khi chạy file ban đầu, nguyên nhân và cách khắc phục:

1. Lỗi UnicodeDecodeError: 'charmap' codec can't decode byte ...
   - Nguyên nhân: File notebook gọi "from ipynb.fs.defs.preprocessing import get_data". Trên hệ điều hành Windows, thư viện "ipynb" khi đọc file notebook khác mặc định sử dụng bảng mã hệ thống (CP1252 / charmap) thay vì UTF-8. Nếu trong file notebook tiền xử lý có chứa ký tự tiếng Việt hoặc ký tự đặc biệt, lệnh load JSON của thư viện sẽ bị sập ngay lập tức.
   - Cách sửa: Không phụ thuộc vào "ipynb.fs.defs". Toàn bộ hàm tiền xử lý và nạp dữ liệu get_data() được chuyển vào file Python thuần túy "util.py" hoặc viết trực tiếp trong file mã nguồn. Ngoài ra, các notebook được lưu theo chuẩn UTF-8 / ASCII escape để tránh lỗi mở file trên Windows.

2. Lỗi NameError: name 'cb' is not defined và name 'thresholds' is not defined
   - Nguyên nhân: 
     + Biến 'cb' bị lỗi do Cell 1 phía trên gặp lỗi UnicodeDecodeError nên lệnh "import catboost as cb" chưa từng được nạp vào bộ nhớ. Khi bấm chạy các cell phía dưới, Python sẽ báo NameError: name 'cb' is not defined.
     + Biến 'thresholds' bị lỗi khi người dùng chạy nhảy cóc cell hoặc cell tính toán dò ngưỡng chưa hoàn tất nhưng đã thực thi cell vẽ đồ thị ở bên dưới.
   - Cách sửa:
     + Khắc phục triệt để lỗi nạp dữ liệu ở Cell 1. Đồng thời khai báo import trực tiếp "import catboost as cb", "import lightgbm as lgb", "import xgboost as xgb" ngay tại cell huấn luyện.
     + Bổ sung cơ chế tự khởi tạo dự phòng cho 'thresholds' và 'opt_threshold' ngay trong cell vẽ đồ thị nếu biến chưa tồn tại trong bộ nhớ.

3. Lỗi chỉ số PaySim thấp (PR-AUC chỉ đạt ~0.29 thay vì ~0.86 như trên Colab)
   - Nguyên nhân: 
     + Sử dụng tham số sample_size nhỏ (ví dụ 100,000 dòng ngẫu nhiên) làm phá vỡ các đặc trưng chuỗi thời gian tích lũy của PaySim (như đếm số lần giao dịch tới tài khoản đích, giao dịch trùng bước thời gian).
     + Sử dụng trọng số phạt mất cân bằng scale_pos_weight quá lớn cho mô hình cây phức tạp khiến mô hình đoán nhầm rất nhiều giao dịch thường thành gian lận (Recall cao ~99% nhưng Precision tụt xuống ~9%, kéo PR-AUC xuống thấp).
   - Cách sửa: 
     + Chạy trên toàn bộ dữ liệu 2.77 triệu dòng PaySim (lọc 2 loại giao dịch có gian lận là TRANSFER và CASH_OUT, loại bỏ 4 cột số dư để chống rò rỉ dữ liệu).
     + Áp dụng mô hình "XGBoost No Scale" với tham số tree_method="hist". Thời gian huấn luyện trên CPU máy tính chỉ mất khoảng 30 giây và tái lập chính xác 100% kết quả trên Colab (PR-AUC đạt 0.8642, F1-Score đạt 0.8338, ROC-AUC đạt 0.9964).

4. Lỗi VS Code hiện chữ M, dấu chấm tròn và lỗi "Could not register service worker: InvalidStateError"
   - Nguyên nhân: 
     + Dấu chấm tròn bên cạnh tên file và chữ M thể hiện tab notebook trong VS Code đang có nội dung chỉnh sửa chưa lưu trong bộ nhớ đệm, dẫn đến xung đột khi file trên ổ đĩa được cập nhật từ bên ngoài.
     + Thông báo "Could not register service worker" xuất hiện khi tiến trình nền (Extension Host) của VS Code bị crash nhiều lần liên tiếp, làm tê liệt giao diện hiển thị webview của Jupyter Notebook.
   - Cách sửa:
     + Đóng các tab notebook có dấu chấm tròn (chọn Don't Save để tránh ghi đè nội dung cũ lên ổ đĩa).
     + Nhấn tổ hợp phím Ctrl + Shift + P, gõ và chọn lệnh "Developer: Reload Window" để khởi động lại bộ nhớ đệm ServiceWorker của VS Code.
     + Mỗi thư mục đều đã được bổ sung file mã nguồn Python độc lập tương ứng (optimal_method.py). Có thể chạy trực tiếp bằng dòng lệnh terminal (ví dụ: py -3.13 notebooks/paysim/optimal_method.py) mà không cần mở qua giao diện Jupyter của VS Code.

5. Lỗi Microsoft Visual C++ 14.0 or greater is required khi build wheel cho greenlet / daal:
   - Nguyên nhân: Máy Windows chưa cài C++ Build Tools nên pip cố gắng compile mã nguồn C/C++.
   - Cách sửa: Ép pip chỉ tải bản pre-compiled binary (wheel đã build sẵn):
     ```bash
     pip install --only-binary :all: greenlet
     # hoặc ép tải binary cho tất cả packages:
     pip install --only-binary :all: -r requirements.txt
     ```

6. Lỗi ModuleNotFoundError: No module named 'sklearnex':
   - Nguyên nhân: File baseline.ipynb, threshold.ipynb gọi from sklearnex import patch_sklearn từ thư viện Intel Extension for Scikit-learn.
   - Cách sửa:
     ```bash
     pip install scikit-learn-intelex
     ```

7. Lỗi FileNotFoundError đối với đường dẫn dữ liệu CSV:
   - Nguyên nhân: Sử dụng đường dẫn tương đối ../../data phụ thuộc vào thư mục làm việc hiện tại khi chạy notebook.
   - Cách sửa: Cập nhật hàm nạp dữ liệu tự động dò tìm đường dẫn dữ liệu qua danh sách các vị trí khả dĩ ('../../data', '../data', 'data') để chạy từ bất kỳ thư mục con nào cũng tìm thấy file CSV.

---

### 3. Tải bộ dữ liệu

Chạy file notebook [notebooks/kaggle.ipynb](file:///notebooks/kaggle.ipynb) hoặc chạy bằng lệnh:
```bash
python -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud', output_dir='data/credit_card'); kagglehub.dataset_download('ealaxi/paysim1', output_dir='data/paysim')"
```

---

## 📂 Cấu trúc Repository & Danh mục các phương pháp

```text
Nhom01_GDG/
├── data/                                # Chứa các bộ dữ liệu CSV (credit_card, paysim, ieee)
├── notebooks/
│   ├── credit_card/                     # Bộ dữ liệu Credit Card Fraud Detection
│   │   ├── preprocessing.ipynb          # Pipeline nạp và chia tập train/val/test
│   │   ├── baseline.ipynb               # Mô hình cơ sở Logistic Regression
│   │   ├── oversampling.ipynb           # Resampling: RandomOverSampler, SMOTE, ADASYN
│   │   ├── undersampling.ipynb          # Resampling: RandomUnderSampler, ClusterCentroids, NearMiss
│   │   ├── hybrid_resampling.ipynb      # [MỚI] Hybrid: SMOTE-Tomek, SMOTE-ENN
│   │   ├── svm.ipynb                    # Support Vector Machine (SVC)
│   │   ├── threshold.ipynb              # Tinh chỉnh ngưỡng phân lớp (Threshold Tuning)
│   │   ├── tree_based.ipynb             # Tree-based: Random Forest, XGBoost, LightGBM, CatBoost
│   │   ├── hyperparameter.ipynb         # Tối ưu hóa siêu tham số bằng Optuna
│   │   └── optimal_method.ipynb         # [MỚI] PHƯƠNG PHÁP TỐI ƯU: Tri-Model Ensemble + Cost Threshold + SHAP
│   │
│   ├── paysim/                          # Bộ dữ liệu PaySim Mobile Money Fraud
│   │   ├── preprocessing.ipynb          # Tiền xử lý & Feature Engineering (Quy tắc No-Balance leakage)
│   │   ├── 01_baseline.ipynb            # [MỚI] Baseline Logistic Regression & Class Weight
│   │   ├── 02_tree_based.ipynb          # [MỚI] Random Forest, XGBoost, LightGBM với scale_pos_weight
│   │   └── optimal_method.ipynb         # [MỚI] Ensemble + Cost-sensitive Threshold + SHAP
│   │
│   ├── ieee/                            # Bộ dữ liệu IEEE-CIS Fraud Detection
│   │   ├── 01_baseline.ipynb            # Baseline LightGBM & ColumnTransformer
│   │   ├── 02_no_feature_engineering.ipynb
│   │   ├── 03_feature_engineering.ipynb # Feature Engineering sâu (UIDs, frequency, group stats)
│   │   ├── lbgm.ipynb                   # K-Fold LightGBM
│   │   └── optimal_method.ipynb         # [MỚI] Optimal Ensemble trên đặc trưng đa chiều
│   └── kaggle.ipynb                     # Tải dữ liệu từ Kaggle Hub
│
├── BAO_CAO_THUC_NGHIEM_FRAUD_DETECTION.md # Báo cáo phân tích chuyên sâu chi tiết (Toàn bộ 11 phương pháp)
├── run_all_benchmarks.py                # Script chạy tự động đánh giá 11 phương pháp trên 3 bộ dữ liệu
└── requirements.txt                     # Danh sách thư viện phụ thuộc
```

---

## 🏆 Tóm tắt kết quả & Phương pháp tối ưu nhất (Optimal Method)

### 1. Vì sao Accuracy gây hiểu lầm? (Accuracy Paradox)
Trên tập Credit Card ($85,443$ mẫu test, $148$ ca gian lận):
- Một mô hình ngây thơ chỉ đoán toàn bộ là bình thường đạt **Accuracy = 99.827%**.
- Nhưng **Recall = 0.00%**, bỏ sót $100\%$ các vụ gian lận, gây thiệt hại ước tính **\$29,600** trong một đợt kiểm tra.
- **Thang đo bắt buộc:** **PR-AUC (Precision-Recall AUC)**, **Recall**, **Precision** và **F1-Score**.

### 2. Phương pháp tối ưu toàn diện nhất
Mô hình **Tri-Model Soft Voting Ensemble (XGBoost + LightGBM + CatBoost) kết hợp Cost-Sensitive Threshold Tuning**:
- Sử dụng trọng số phạt `scale_pos_weight` thay vì resampling đơn thuần (tránh làm mất cân đối phân phối gốc).
- Hòa trộn xác suất 3 kiến trúc: $P_{\text{ensemble}} = 0.4 \cdot P_{\text{XGB}} + 0.4 \cdot P_{\text{LGB}} + 0.2 \cdot P_{\text{CatBoost}}$.
- Tối ưu hóa ngưỡng quyết định theo chi phí thực tế: $\mathcal{L} = 200 \times \text{FN} + 10 \times \text{FP}$.
- **Hiệu quả:** Đạt **PR-AUC cao nhất trên cả 3 bộ dữ liệu** ($0.8166$ Credit Card, $0.9909$ PaySim, $0.8133$ IEEE-CIS) và **tiết kiệm hơn 81% chi phí tổn thất** so với mô hình cơ sở.
