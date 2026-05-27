# 💳 HỆ THỐNG DỰ BÁO RỦI RO TÍN DỤNG BẰNG THUẬT TOÁN HỌC MÁY
> **Môn học:** Học máy ứng dụng | **Đơn vị:** Đại học Kinh tế TP.HCM (UEH)

---

## 👥 1. NHÓM THỰC HIỆN (NHÓM 5 - NGÀNH KỸ THUẬT PHẦN MỀM)
* **Trần Văn Ngân** (Trưởng nhóm)
* **Huỳnh Nhật Gia Lạc**
* **Nguyễn Hoàng Minh** 
* **Phạm Thành Nhân** 
* **Hồ Xuân Lộc** 

---

## 🛠️ 2. QUY TRÌNH TRIỂN KHAI PHÒNG CHỐNG DATA LEAKAGE (PIPELINE)

Hệ thống được thiết kế theo quy trình khép kín nhằm bảo vệ tính khách quan của dữ liệu:
1.  **EDA & Data Quality:** Nhận diện kiểu biến, thống kê mô tả, tỷ lệ thiếu, dòng trùng lặp, miền giá trị hợp lệ, histogram/boxplot và IQR outlier report.
2.  **Làm sạch dữ liệu:** Loại dòng trùng lặp và chỉ loại outlier bất khả lý theo ngữ cảnh nghiệp vụ (`person_age` > 100, `person_emp_length` > 60), không xóa hàng loạt các giá trị cao nhưng có thể hợp lệ như thu nhập hoặc khoản vay.
3.  **Chia tách dữ liệu:** Thực hiện chia tập Train/Test tỷ lệ 80/20 với `stratify=y` trước khi điền khuyết (Imputation) nhằm triệt tiêu rủi ro rò rỉ thông tin tập Test. Với LightGBM, tách thêm validation set từ Train để early stopping; Test chỉ dùng cho đánh giá cuối.
4.  **Phân nhánh bệ phóng cho từng mô hình:**
    * *Kiến trúc LightGBM:* Giữ nguyên `NaN`, ép kiểu chữ thô sang kiểu danh mục (`category`).
    * *Kiến trúc Random Forest:* Thêm missing indicators, điền khuyết theo Median tập Train và áp dụng One-Hot Encoding.
    * *Kiến trúc Logistic Regression:* Tiếp quản dữ liệu mã hóa của RF và Chuẩn hóa phân phối (`StandardScaler`).

---

## 📊 3. KẾT QUẢ THỰC NGHIỆM ĐỐI CHIẾU (TẬP TEST)

Bảng tổng hợp năng lực dự báo thực tế đối với nhóm mục tiêu **Vỡ nợ (Nhãn 1)**:

| Mô hình | Accuracy | Precision | Recall | F1-score | AUC-ROC | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.62% | 53.94% | 78.14% | 63.82% | 0.8718 | **0.039s** |
| **Random Forest** | 91.82% | 85.46% | 75.46% | 80.15% | 0.9319 | 0.847s |
| **LightGBM (Champion)** | **92.97%** | **88.30%** | **78.21%** | **82.95%** | **0.9479** | 1.342s |

### 💡 Biện giải cốt lõi từ dữ liệu thực tế:
* **Logistic Regression (Baseline):** Tốc độ nhanh nhất, Recall tương đối cao ($78.14\%$) nhưng Precision thấp ($53.94\%$), gây tỷ lệ báo động giả lớn.
* **Random Forest:** Precision tốt ($85.46\%$), nhưng Recall thấp hơn LightGBM ($75.46\%$), nên vẫn có nguy cơ bỏ sót một phần hồ sơ vỡ nợ.
* **LightGBM (Mô hình tối ưu):** Nhờ cơ chế mở rộng số lá (`num_leaves=63`) kết hợp trọng số phạt lệch nhãn (`scale_pos_weight`), LightGBM đạt **F1-score tốt nhất $82.95\%$** và **AUC-ROC $0.9479$**. Mô hình cân bằng tốt nhất giữa Precision ($88.30\%$) và Recall ($78.21\%$) trong thí nghiệm.

---

## 👁️ 4. GIẢI THÍCH MÔ HÌNH VỚI SHAP VALUE (XAI)
Để mở "hộp đen" thuật toán, nhóm ứng dụng kỹ thuật giải thích Học máy hiện đại:
* **Summary Plot (Toàn cục):** Chứng minh tỷ lệ khoản vay trên thu nhập (`loan_percent_income`) và lãi suất (`loan_int_rate`) càng cao (chấm đỏ) càng đẩy dự báo về phía Vỡ nợ.
* **Waterfall Plot (Cục bộ):** Trực quan hóa chi tiết lộ trình cộng/trừ điểm rủi ro dựa trên các chỉ số cá thể để minh bạch hóa lý do phê duyệt hoặc từ chối của một hồ sơ cụ thể.

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY MÃ NGUỒN
1.  **Cài đặt thư viện:** `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jinja2`
2.  **Khởi chạy:** Mở tệp `CreditRiskPrediction.ipynb` trên Jupyter Notebook và chọn `Run All`. Môi trường yêu cầu có sẵn tệp `Data/credit_risk_dataset.csv` để nạp Pipeline tự động.
