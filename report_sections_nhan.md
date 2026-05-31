# Gợi ý chỉnh phần Nhân: chỉ bổ sung chỗ còn thiếu

Không cần viết lại toàn bộ mục 3.2 và 3.4. Phần hiện tại trong báo cáo đã đúng hướng, chỉ nên bổ sung thêm số liệu cụ thể, hình minh họa và vài đoạn giải thích cho tự nhiên hơn. Các hình đã tạo sẵn trong `report_assets/`.

## Bản đồ chèn nhanh

Nếu đang sửa trực tiếp Google Docs, làm theo thứ tự này là đủ:

1. **Ở mục 3.2, sau bảng mô tả các biến dữ liệu**  
   Chèn đoạn về `32,581 dòng`, `165 dòng trùng`, missing của `person_emp_length` và `loan_int_rate`.  
   Có thể chèn thêm hình `eda_preprocess_summary_table.png` ngay sau đoạn này.

2. **Ở mục 3.2, sau đoạn thống kê mô tả hoặc trước phần biểu đồ phân phối**  
   Chèn đoạn nhận xét mean/median: `person_income` và `loan_amnt` bị lệch phải.  
   Chèn code `desc_stats` ngay dưới đoạn này nếu muốn minh họa cách tính.

3. **Ở mục 3.2, ngay sau phần nói về biểu đồ phân phối**  
   Chèn hình `eda_numeric_histograms.png`.  
   Không cần viết dài, chỉ cần caption: “Hình 3.1. Histogram phân phối các biến số chính”.

4. **Ở mục 3.2, ngay sau phần nói về outlier/boxplot**  
   Chèn đoạn giải thích: không xóa toàn bộ IQR outlier, chỉ xóa `person_age > 100` và `person_emp_length > 60`.  
   Chèn hình `eda_numeric_boxplots.png` ngay sau đoạn này.

5. **Cuối mục 3.2**  
   Chèn câu kết luận EDA: dữ liệu có duplicate, missing, outlier bất hợp lý và mất cân bằng lớp khoảng `21.82%` vỡ nợ, nên cần `stratify=y` và xử lý missing cẩn thận.

6. **Ở mục 3.4, đầu phần tiền xử lý**  
   Chèn đoạn về việc loại `165` duplicate và `7` outlier bất hợp lý.  
   Code đi kèm là đoạn `df_clean = df.drop_duplicates().copy()` và filter domain-rule.
   Nếu trong báo cáo đang có hình “Biểu đồ Boxplot phát hiện giá trị ngoại lai của person_age và person_emp_length”, nên thay bằng hình `eda_numeric_boxplots.png` để khớp với cell boxplot hiện có trong notebook. Hình này thể hiện đầy đủ các biến số chính, sau đó phần chữ giải thích rõ nhóm chỉ loại `person_age > 100` và `person_emp_length > 60`.

7. **Ở mục 3.4, sau đoạn chia train/test**  
   Chèn đoạn chống data leakage: chia train/test trước imputation, dùng `stratify=y`, train `25,927` dòng và test `6,482` dòng.  
   Code đi kèm là đoạn `train_test_split(...)`.

8. **Ở mục 3.4, sau đoạn xử lý missing values**  
   Chèn đoạn phân biệt LightGBM với RF/LR: LightGBM giữ `NaN`, RF/LR thêm missing indicator và median imputation từ train.  
   Code đi kèm là vòng lặp `for col in ["loan_int_rate", "person_emp_length"]`.

9. **Ở mục 3.4, sau đoạn mã hóa dữ liệu**  
   Chèn đoạn: RF/LR dùng One-Hot Encoding, riêng Logistic Regression dùng StandardScaler; LightGBM không cần scale.

10. **Cuối mục 3.4**  
    Chèn câu kết: pipeline phù hợp vì không xóa outlier tài chính hợp lệ, tránh data leakage, và chuẩn bị dữ liệu riêng cho từng mô hình.

Phần có thể bỏ nếu báo cáo quá dài: code `desc_stats`, code IQR đầy đủ, hoặc chèn lại bảng `eda_preprocess_summary_table.png` lần thứ hai ở mục 3.4. Phần nên giữ chắc chắn: missing/duplicate, histogram, boxplot, đoạn không xóa toàn bộ IQR outlier, và đoạn chống data leakage.

## 1. Mục 3.2 - EDA

**Giữ lại phần mô tả dataset và bảng biến hiện có.**  
Sau phần mô tả các biến, bổ sung đoạn ngắn dưới đây để phần EDA có số liệu rõ hơn:

> Khi kiểm tra chất lượng dữ liệu ban đầu, tập `credit_risk_dataset` có 32,581 dòng và 12 cột. Trong đó có 165 dòng trùng lặp, cần loại bỏ trước khi chia train/test để tránh cùng một hồ sơ xuất hiện ở cả hai tập. Dữ liệu chỉ bị thiếu ở hai biến: `person_emp_length` thiếu 895 dòng, tương đương 2.75%, và `loan_int_rate` thiếu 3,116 dòng, tương đương 9.56%. Các biến còn lại không có missing values, nên nhóm không cần loại bỏ cột nào ở bước EDA.

Chèn bảng/hình này ngay sau đoạn trên:

`report_assets/eda_preprocess_summary_table.png`

Code minh họa đặt ngay dưới phần kiểm tra missing/duplicate:

```python
missing_report = df.isna().sum()
missing_report = missing_report[missing_report > 0]

duplicate_count = df.duplicated().sum()

print(missing_report)
print("Duplicate rows:", duplicate_count)
```

Sau phần thống kê mô tả, thêm đoạn này:

> Các biến tài chính trong bộ dữ liệu có xu hướng lệch phải. Ví dụ, `person_income` có trung bình khoảng 66,074.85 nhưng trung vị chỉ 55,000; `loan_amnt` có trung bình 9,589.37 nhưng trung vị 8,000. Điều này khá hợp lý với dữ liệu tín dụng, vì một nhóm nhỏ khách hàng có thu nhập hoặc khoản vay rất cao có thể kéo giá trị trung bình tăng lên. Vì vậy, nhóm không xử lý outlier một cách máy móc chỉ dựa trên IQR.

Code minh họa đặt ngay dưới phần thống kê mô tả:

```python
desc_stats = df[num_cols].describe().T
desc_stats["median"] = df[num_cols].median()
desc_stats["iqr"] = desc_stats["75%"] - desc_stats["25%"]
display(desc_stats.round(3))
```

Chèn hình histogram sau đoạn nói về phân phối:

`report_assets/eda_numeric_histograms.png`

Sau phần boxplot/outlier, bổ sung đoạn này:

> Boxplot cho thấy nhiều biến có điểm nằm ngoài râu hộp, đặc biệt là `loan_amnt`, `person_income`, `person_age` và `person_emp_length`. Tuy nhiên, trong bối cảnh tín dụng, các giá trị như thu nhập cao hoặc khoản vay lớn chưa chắc là lỗi dữ liệu. Vì vậy, nhóm chỉ xem các giá trị bất hợp lý về mặt thực tế là outlier cần loại bỏ, cụ thể là `person_age > 100` và `person_emp_length > 60`.

Chèn hình boxplot ngay sau đoạn trên:

`report_assets/eda_numeric_boxplots.png`

Code minh họa đặt trong phần phát hiện outlier:

```python
q1 = df[num_cols].quantile(0.25)
q3 = df[num_cols].quantile(0.75)
iqr = q3 - q1

iqr_outlier_mask = (df[num_cols].lt(q1 - 1.5 * iqr)) | (
    df[num_cols].gt(q3 + 1.5 * iqr)
)

domain_outliers = {
    "person_age > 100": (df["person_age"] > 100).sum(),
    "person_emp_length > 60": (
        df["person_emp_length"].notna() & (df["person_emp_length"] > 60)
    ).sum(),
}
print(domain_outliers)
```

**Câu kết nên thêm vào cuối mục 3.2:**

> Từ EDA, nhóm xác định ba vấn đề cần xử lý trước khi huấn luyện mô hình: dòng trùng lặp, missing values ở `person_emp_length` và `loan_int_rate`, cùng một số outlier bất hợp lý theo miền dữ liệu. Ngoài ra, tỷ lệ vỡ nợ khoảng 21.82% cho thấy dữ liệu có mất cân bằng lớp, nên ở bước mô hình hóa cần dùng `stratify=y` khi chia train/test và cân nhắc trọng số lớp cho LightGBM.

## 2. Mục 3.4 - Tiền xử lý dữ liệu

Phần 3.4 hiện đúng ý nhưng nên viết lại gọn hơn, tránh cảm giác chung chung. Có thể thay phần mô tả tiền xử lý hiện tại bằng đoạn sau:

> Sau EDA, nhóm thực hiện tiền xử lý theo hướng giữ lại tối đa thông tin thật của dữ liệu. Trước hết, 165 dòng trùng lặp được loại bỏ. Tiếp theo, nhóm chỉ loại các outlier bất hợp lý theo miền dữ liệu, gồm `person_age > 100` và `person_emp_length > 60`, tổng cộng 7 dòng. Các giá trị cao ở thu nhập hoặc khoản vay không bị xóa hàng loạt vì vẫn có thể phản ánh hồ sơ vay hợp lệ.

Code đặt ngay sau đoạn xử lý duplicate/outlier:

```python
df_clean = df.drop_duplicates().copy()

df_clean = df_clean[
    (df_clean["person_age"] <= 100) &
    (
        df_clean["person_emp_length"].isna() |
        (df_clean["person_emp_length"] <= 60)
    )
].copy()
```

Chèn hình này ngay sau đoạn xử lý duplicate/outlier để khớp với notebook:

`report_assets/eda_numeric_boxplots.png`

Caption nên chỉnh lại cho đúng nội dung hình:

> Hình . Biểu đồ Boxplot kiểm tra giá trị ngoại lai của các biến số chính

Thêm đoạn chống data leakage:

> Sau khi làm sạch, dữ liệu còn 32,409 dòng. Nhóm chia train/test theo tỷ lệ 80/20 và dùng `stratify=y` để giữ tỷ lệ vỡ nợ ổn định giữa hai tập. Việc chia dữ liệu được thực hiện trước bước imputation, vì nếu tính median trên toàn bộ dữ liệu thì thông tin từ tập test sẽ vô tình đi vào quá trình huấn luyện.

Code đặt ngay sau đoạn chia train/test:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Thêm đoạn xử lý missing theo từng mô hình:

> Với LightGBM, nhóm giữ nguyên `NaN` và chuyển các biến phân loại sang kiểu `category`, vì LightGBM có khả năng học hướng tách phù hợp cho missing values. Với Logistic Regression và Random Forest, nhóm thêm hai biến đánh dấu missing cho `loan_int_rate` và `person_emp_length`, sau đó điền khuyết bằng median tính từ tập train. Cách này vừa loại bỏ `NaN` cho các mô hình không xử lý trực tiếp missing values, vừa giữ lại thông tin rằng một giá trị ban đầu từng bị thiếu.

Code đặt ngay dưới đoạn xử lý missing:

```python
for col in ["loan_int_rate", "person_emp_length"]:
    X_train_imputed[f"{col}_missing"] = X_train_imputed[col].isna().astype(int)
    X_test_imputed[f"{col}_missing"] = X_test_imputed[col].isna().astype(int)

    median_value = X_train_imputed[col].median()
    X_train_imputed[col] = X_train_imputed[col].fillna(median_value)
    X_test_imputed[col] = X_test_imputed[col].fillna(median_value)
```

Thêm đoạn mã hóa/chuẩn hóa:

> Sau imputation, các biến phân loại được One-Hot Encoding cho Logistic Regression và Random Forest. Riêng Logistic Regression được chuẩn hóa bằng StandardScaler vì mô hình tuyến tính nhạy với thang đo của biến số. Random Forest và LightGBM không cần chuẩn hóa vì hai mô hình dựa trên cây quyết định.

Nếu muốn có thêm một bảng kiểm tra sau preprocess, chèn lại hình:

`report_assets/eda_preprocess_summary_table.png`

**Câu kết nên thêm vào cuối mục 3.4:**

> Như vậy, pipeline tiền xử lý vừa phù hợp với lý thuyết tiền xử lý dữ liệu, vừa bám sát đặc thù của dataset credit risk. Nhóm không xóa dữ liệu cực đoan một cách đại trà, tránh data leakage bằng cách chia train/test trước khi impute, và chuẩn bị dữ liệu riêng cho từng nhóm mô hình thay vì dùng một cách xử lý chung cho tất cả.

## 3. Nhận xét nhanh về phần Nhân

Phần Nhân hiện **đúng hướng**, nhưng nên bổ sung các nội dung trên để đầy đủ hơn:

- Có số liệu missing/duplicate rõ ràng.
- Có thống kê mô tả và nhận xét phân phối lệch phải.
- Có histogram và boxplot từ dataset thật.
- Có giải thích vì sao không xóa toàn bộ IQR outlier.
- Có mô tả chống data leakage khi chia train/test trước imputation.
- Code được đặt theo từng bước nhỏ, không nhét một khối code dài vào cuối mục.
