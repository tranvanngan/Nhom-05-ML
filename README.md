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
1.  **EDA & Lọc Outliers:** Phát hiện lỗi hệ thống bằng Boxplot (`person_age` > 100, `person_emp_length` > 60) và loại bỏ. Xác định `loan_percent_income` và `loan_grade` có độ tương quan gốc cao nhất với mục tiêu.
2.  **Chia tách dữ liệu:** Thực hiện chia tập Train/Test tỷ lệ 80/20 với `stratify=y` trước khi điền khuyết (Imputation) nhằm triệt tiêu hoàn toàn rủi ro rò rỉ thông tin tập Test.
3.  **Phân nhánh bệ phóng cho từng mô hình:**
    * *Kiến trúc LightGBM:* Giữ nguyên `NaN`, ép kiểu chữ thô sang kiểu danh mục (`category`).
    * *Kiến trúc Random Forest:* Điền khuyết theo Median tập Train và áp dụng One-Hot Encoding.
    * *Kiến trúc Logistic Regression:* Tiếp quản dữ liệu mã hóa của RF và Chuẩn hóa phân phối (`StandardScaler`).

---

## 📊 3. KẾT QUẢ THỰC NGHIỆM ĐỐI CHIẾU (TẬP TEST)

Bảng tổng hợp năng lực dự báo thực tế đối với nhóm mục tiêu **Vỡ nợ (Nhãn 1)**:

| Mô hình | Accuracy | Precision | Recall | F1-score | AUC-ROC | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 81.28% | 54.53% | **78.97%** | 64.51% | 0.8765 | **0.241s** |
| **Random Forest** | 92.36% | 87.94% | 74.80% | 80.84% | 0.9352 | 2.403s |
| **LightGBM (Champion)** | **93.18%** | **88.46%** | **78.61%** | **83.24%** | **0.9503** | 3.457s |

### 💡 Biện giải cốt lõi từ dữ liệu thực tế:
* **Logistic Regression (Baseline):** Tốc độ nhanh nhất, Recall cao ($78.97\%$) nhưng Precision quá thấp ($54.53\%$), gây tỷ lệ báo động giả lớn (từ chối nhầm nhiều khách hàng tốt).
* **Random Forest:** Bộ lọc chuẩn xác (Precision $87.94\%$) nhưng độ bao phủ kém (Recall tụt xuống $74.80\%$), dễ làm lọt lưới nợ xấu gây mất vốn gốc ngân hàng.
* **LightGBM (Mô hình tối ưu):** Nhờ cơ chế mở rộng số lá (`num_leaves=63`) kết hợp trọng số phạt lệch nhãn (`scale_pos_weight`), LightGBM đạt điểm **F1-score tối quý $83.24\%$** và **AUC-ROC $0.9503$**. Mô hình dung hòa hoàn hảo: vừa giữ bộ lọc siêu chuẩn ($88.46\%$), vừa bao phủ trọn vẹn rủi ro ($78.61\%$).

---

## 👁️ 4. GIẢI THÍCH MÔ HÌNH VỚI SHAP VALUE (XAI)
Để mở "hộp đen" thuật toán, nhóm ứng dụng kỹ thuật giải thích Học máy hiện đại:
* **Summary Plot (Toàn cục):** Chứng minh tỷ lệ khoản vay trên thu nhập (`loan_percent_income`) và lãi suất (`loan_int_rate`) càng cao (chấm đỏ) càng đẩy xác suất dự báo về phía Vỡ nợ.
* **Waterfall Plot (Cục bộ):** Trực quan hóa chi tiết lộ trình cộng/trừ điểm rủi ro dựa trên các chỉ số cá thể để minh bạch hóa lý do phê duyệt hoặc từ chối của một hồ sơ cụ thể.

---

## 🚀 5. HƯỚNG DẪN CÀI ĐẶT & CHẠY MÃ NGUỒN
1.  **Cài đặt thư viện:** `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jinja2`
2.  **Khởi chạy:** Mở tệp `credit_risk_scoring.ipynb` trên Jupyter Notebook và chọn `Run All`. Môi trường yêu cầu có sẵn tệp `credit_risk_dataset.csv` tại thư mục gốc để nạp Pipeline tự động.
