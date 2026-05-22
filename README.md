# Machine Learning Project – Sales & Customer Behaviour Insights

Dự án sử dụng bộ dữ liệu bán hàng gồm ba bảng chính: thông tin giao dịch, thông tin khách hàng và thông tin sản phẩm. Mục tiêu của project là chuẩn bị dữ liệu sạch để phục vụ các bước phân tích và xây dựng mô hình học máy ở giai đoạn sau.

Ở giai đoạn hiện tại, repo tập trung vào phần **data preprocessing / data cleaning**. Phần này chỉ làm sạch từng bảng dữ liệu riêng lẻ, chưa thực hiện join bảng, chưa tạo feature liên bảng và chưa huấn luyện mô hình.

## Dataset

Dữ liệu đầu vào gồm ba file CSV:

```text
data/raw/
├── sales_data.csv
├── customer_info.csv
└── product_info.csv
```

Ý nghĩa từng bảng:

| File | Nội dung |
|---|---|
| `sales_data.csv` | Dữ liệu giao dịch bán hàng, gồm mã đơn hàng, khách hàng, sản phẩm, số lượng, giá, ngày đặt hàng, trạng thái giao hàng, phương thức thanh toán, khu vực và chiết khấu. |
| `customer_info.csv` | Thông tin khách hàng, gồm mã khách hàng, email, ngày đăng ký, giới tính, khu vực và hạng thành viên. |
| `product_info.csv` | Thông tin sản phẩm, gồm mã sản phẩm, tên sản phẩm, danh mục, ngày ra mắt, giá gốc và mã nhà cung cấp. |

## Preprocessing scope

Các bước đã thực hiện trong phần preprocessing:

- Đọc ba bảng dữ liệu gốc.
- Kiểm tra và xử lý giá trị thiếu.
- Chuẩn hóa dữ liệu dạng chữ và các nhãn phân loại bị sai/không nhất quán.
- Chuyển đổi kiểu dữ liệu số và ngày tháng.
- Kiểm tra outlier bằng phương pháp IQR.
- Thống kê mô tả cho các biến số chính.
- Vẽ histogram và boxplot trong notebook.
- Xuất ba file dữ liệu đã clean.

## Folder structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   │   ├── sales_data.csv
│   │   ├── customer_info.csv
│   │   └── product_info.csv
│   └── processed/
│       ├── cleaned_sales_data.csv
│       ├── cleaned_customer_info.csv
│       └── cleaned_product_info.csv
├── notebooks/
│   └── 01_data_preprocessing.ipynb
└── src/
    └── preprocess_clean_data.py
```

## File description

| File | Công dụng |
|---|---|
| `notebooks/01_data_preprocessing.ipynb` | Notebook trình bày quá trình preprocessing: đọc dữ liệu, kiểm tra missing values, thống kê mô tả, kiểm tra outlier, vẽ histogram/boxplot và xuất dữ liệu clean. |
| `src/preprocess_clean_data.py` | Script Python dùng để chạy lại pipeline làm sạch dữ liệu từ raw CSV và xuất ra các file clean. |
| `data/processed/cleaned_sales_data.csv` | Bảng giao dịch sau khi làm sạch. |
| `data/processed/cleaned_customer_info.csv` | Bảng khách hàng sau khi làm sạch. |
| `data/processed/cleaned_product_info.csv` | Bảng sản phẩm sau khi làm sạch. |

## How to run

Cài các thư viện cần thiết:

```bash
pip install pandas matplotlib jupyter
```

Chạy script preprocessing từ thư mục gốc của repo:

```bash
python src/preprocess_clean_data.py
```

Script sẽ đọc dữ liệu từ:

```text
data/raw/
```

và xuất dữ liệu đã làm sạch vào:

```text
data/processed/
```

Để xem thống kê mô tả và biểu đồ phân phối, mở notebook:

```text
notebooks/01_data_preprocessing.ipynb
```

## Output

Sau khi chạy preprocessing, các file chính cần dùng cho bước tiếp theo là:

```text
data/processed/cleaned_sales_data.csv
data/processed/cleaned_customer_info.csv
data/processed/cleaned_product_info.csv
```

Ba file này giữ nguyên cấu trúc bảng riêng biệt để thành viên phụ trách data integration / feature engineering xử lý ở bước sau.

