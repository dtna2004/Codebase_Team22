# Child Mind Institute - Problematic Internet Use

## Mục tiêu
Dự án này nhằm mục đích dự đoán **Severity Impairment Index (SII)** dựa trên dữ liệu về hành vi trực tuyến và các yếu tố liên quan. Cuộc thi này yêu cầu xây dựng mô hình dự đoán chính xác mức độ ảnh hưởng của việc sử dụng internet gây ra vấn đề cho trẻ em và thanh thiếu niên.

### Đánh giá
Kết quả được đánh giá dựa trên **Quadratic Weighted Kappa (QWK)**, một chỉ số đo lường mức độ đồng thuận giữa dự đoán và nhãn thực tế, đặc biệt phù hợp với bài toán phân loại đa lớp có thứ tự.

---

## Các bước chính trong dự án

### 1. Phân tích dữ liệu (Data Exploration)
- **Mục tiêu:** Hiểu rõ cấu trúc dữ liệu, phân bố của các đặc trưng và cột mục tiêu `sii`.
- **Công cụ:** Sử dụng các thư viện như `pandas`, `matplotlib`, `seaborn` để phân tích và trực quan hóa dữ liệu.
- **Kết quả:**
  - Phân bố của cột `sii` (biểu đồ cột).
  - Tỷ lệ dữ liệu bị thiếu (biểu đồ thanh).
  - Phân bố của các đặc trưng quan trọng như **tuổi**, **giới tính**, **BMI**.

---

### 2. Tiền xử lý dữ liệu (Data Preprocessing)
- **Mục tiêu:** Xử lý dữ liệu bị thiếu, chuẩn hóa dữ liệu và xử lý dữ liệu thời gian thực.
- **Công cụ:**
  - `KNN Imputer` để điền các giá trị bị thiếu.
  - `StandardScaler` để chuẩn hóa dữ liệu.
- **Kết quả:**
  - Dữ liệu được làm sạch và chuẩn hóa để sẵn sàng cho quá trình huấn luyện mô hình.

---

### 3. Tạo đặc trưng (Feature Engineering)
- **Mục tiêu:** Tạo các đặc trưng mới dựa trên tương tác giữa các biến và trích xuất đặc trưng từ dữ liệu thời gian thực.
- **Công cụ:**
  - Sử dụng `AutoEncoder` để trích xuất đặc trưng từ dữ liệu thời gian thực.
- **Kết quả:**
  - Các đặc trưng mới như `BMI_Age`, `Internet_Hours_Age`, `BMI_Internet_Hours`, và các tỷ lệ khác.

---

### 4. Huấn luyện mô hình (Model Training)
- **Mục tiêu:** Xây dựng và huấn luyện các mô hình dự đoán `sii`.
- **Công cụ:**
  - Sử dụng các mô hình như `LightGBM`, `XGBoost`, `CatBoost`, và `TabNet`.
- **Kết quả:**
  - Mô hình được huấn luyện với các tham số tối ưu, đạt được kết quả QWK cao trên tập validation.

---

### 5. Ensemble Learning
- **Mục tiêu:** Kết hợp kết quả từ nhiều mô hình để tăng độ chính xác.
- **Công cụ:**
  - Sử dụng `Voting Regressor` để kết hợp các mô hình.
- **Kết quả:**
  - Mô hình tổng hợp đạt được QWK cao hơn so với các mô hình riêng lẻ.

---

### 6. Đánh giá mô hình (Model Evaluation)
- **Mục tiêu:** Đánh giá hiệu suất của mô hình trên tập test.
- **Công cụ:**
  - Sử dụng **Quadratic Weighted Kappa (QWK)** để đánh giá.
- **Kết quả:**
  - Kết quả QWK trên tập test được tối ưu hóa bằng cách điều chỉnh ngưỡng (**Threshold Optimization**).

---

### 7. Dự đoán và nộp kết quả (Prediction and Submission)
- **Mục tiêu:** Dự đoán `sii` trên tập test và nộp kết quả.
- **Công cụ:**
  - Sử dụng mô hình tổng hợp để dự đoán và lưu kết quả vào file `submission.csv`.
- **Kết quả:**
  - File `submission.csv` được nộp lên Kaggle để đánh giá.

