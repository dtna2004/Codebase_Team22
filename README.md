# Child Mind Institute - Problematic Internet Use

## Mục tiêu
Dự án này nhằm mục đích dự đoán **Severity Impairment Index (SII)** dựa trên dữ liệu về hành vi trực tuyến và các yếu tố liên quan. Cuộc thi này yêu cầu xây dựng mô hình dự đoán chính xác mức độ ảnh hưởng của việc sử dụng internet gây ra vấn đề cho trẻ em và thanh thiếu niên.

### Đánh giá
Kết quả được đánh giá dựa trên **Quadratic Weighted Kappa (QWK)**, một chỉ số đo lường mức độ đồng thuận giữa dự đoán và nhãn thực tế, đặc biệt phù hợp với bài toán phân loại đa lớp có thứ tự.

---

## 1. Ứng dụng của các thuật toán học máy và giải thích các lựa chọn

### 1.1. Thuật toán học máy được sử dụng
- **LightGBM**: Một framework gradient boosting hiệu quả và nhanh chóng, đặc biệt phù hợp với các tập dữ liệu lớn. LightGBM được sử dụng để xử lý các tính năng dữ liệu và dự đoán chỉ số suy giảm nghiêm trọng (sii).
- **XGBoost**: Một thuật toán gradient boosting khác, cũng được sử dụng rộng rãi trong các ứng dụng thực tế. XGBoost được sử dụng để tăng cường độ chính xác của mô hình.
- **CatBoost**: Một thuật toán gradient boosting được tối ưu hóa cho các tính năng phân loại. CatBoost được sử dụng để xử lý các tính năng phân loại mà không cần quá nhiều tiền xử lý.
- **TabNet**: Một mô hình học sâu được thiết kế đặc biệt cho các tác vụ hồi quy và phân loại trên dữ liệu có cấu trúc. TabNet được sử dụng để tận dụng sức mạnh của mạng nơ-ron trong việc học các tính năng phức tạp.
- **Voting Regressor**: Một mô hình tập hợp (ensemble) kết hợp các dự đoán từ nhiều mô hình khác nhau (LightGBM, XGBoost, CatBoost, TabNet) để cải thiện độ chính xác và độ ổn định của mô hình.

### 1.2. Giải thích các lựa chọn
#### LightGBM, XGBoost, CatBoost
- **LightGBM**: hiệu quả trong việc xử lý các tính năng số và phân loại, đặc biệt là trong bộ dữ liệu này, nơi có sự kết hợp giữa các tính năng số (như tuổi, BMI, số giờ sử dụng internet) và phân loại (như mùa, giới tính). LightGBM cũng nhanh chóng và hiệu quả trong việc xử lý các tập dữ liệu lớn, điều này rất phù hợp với bộ dữ liệu có nhiều mẫu và tính năng.
- **XGBoost**: sử dụng để tăng cường độ chính xác của mô hình, đặc biệt là trong các tác vụ phân loại và hồi quy. XGBoost có khả năng xử lý tốt các tính năng số và phân loại, đồng thời cung cấp các tùy chọn điều chỉnh siêu tham số linh hoạt, giúp tối ưu hóa hiệu suất trên bộ dữ liệu này.
- **CatBoost**: tối ưu hóa cho các tính năng phân loại, giúp xử lý các tính năng phân loại mà không cần quá nhiều tiền xử lý. Trong bộ dữ liệu này, có nhiều tính năng phân loại như mùa, giới tính, và các tính năng khác, CatBoost giúp tăng độ chính xác bằng cách xử lý các tính năng này một cách hiệu quả.
#### TabNet
- Được chọn để tận dụng khả năng học tính năng phức tạp của mạng nơ-ron, đặc biệt là trong các tác vụ có tính năng đa dạng và phức tạp. Trong bộ dữ liệu này, có nhiều tính năng tương tác phức tạp giữa các biến (ví dụ: tương tác giữa tuổi và số giờ sử dụng internet), TabNet giúp mô hình hóa các tương tác này một cách hiệu quả.

#### Voting Regressor
- kết hợp các mô hình khác nhau, giúp giảm thiểu rủi ro overfitting và cải thiện độ chính xác tổng thể của mô hình. Trong bộ dữ liệu này, sử dụng Voting Regressor giúp tận dụng sức mạnh của nhiều mô hình khác nhau, từ đó tăng cường độ chính xác và độ ổn định của mô hình.

---

## 2. Hiệu suất của mô hình

### 2.1. Số liệu đánh giá
- **Quadratic Weighted Kappa (QWK)**:
  - Mục đích: QWK là một số liệu đánh giá hiệu suất mô hình đặc biệt phù hợp cho các tác vụ phân loại có thứ tự (ordinal classification). Trong bài toán này, QWK được sử dụng để đo lường mức độ đồng thuận giữa các dự đoán của mô hình và các giá trị thực tế của chỉ số suy giảm nghiêm trọng (sii).
  - Tầm quan trọng: QWK không chỉ đo lường độ chính xác mà còn tính đến thứ tự của các lớp (None, Mild, Moderate, Severe). Điều này rất phù hợp với bài toán này vì chỉ số sii là một biến có thứ tự, và việc dự đoán sai một lớp gần với lớp thực tế sẽ được đánh giá cao hơn so với dự đoán sai một lớp xa.
- **Mean Train QWK**:
  - Mục đích: đo lường hiệu suất trung bình của mô hình trên tập huấn luyện qua các fold trong quá trình cross-validation.
  - Tầm quan trọng: Giúp kiểm tra xem mô hình có học tốt trên dữ liệu huấn luyện hay không. Nếu giá trị này quá cao, có thể mô hình đang overfit, nếu quá thấp, mô hình có thể chưa học đủ tốt.
- **Mean Validation QWK**:
  - Mục đích: Hiệu suất trung bình trên tập validation qua cross-validation.
  - Tầm quan trọng: Đánh giá khả năng tổng quát hóa của mô hình.
- **Optimized QWK Score**:
  - Mục đích: Hiệu suất sau khi tối ưu hóa ngưỡng, cải thiện độ chính xác phân loại.

### 2.2. Kết quả
- **Mean Train QWK**: 0.7479
- **Mean Validation QWK**: 0.4880
- **Optimized QWK Score**: 0.531

---

## 3. Cải tiến mô hình

### 3.1. Điều chỉnh siêu tham số
#### LightGBM
- **learning_rate**: 0.046 (học chậm hơn nhưng ổn định hơn).
- **max_depth**: 12 (khám phá tính năng phức tạp).
- **num_leaves**: 478 (phân loại chi tiết hơn).
- **feature_fraction**: 0.893 (giảm overfitting).
- **bagging_fraction**: 0.784 (giảm overfitting).
- **lambda_l1**: 10, **lambda_l2**: 0.01 (giảm độ phức tạp mô hình).

#### XGBoost
- **learning_rate**: 0.05.
- **max_depth**: 6.
- **n_estimators**: 200.
- **subsample, colsample_bytree**: 0.8 (giảm overfitting).
- **reg_alpha, reg_lambda**: 1, 5.

#### CatBoost
- **learning_rate**: 0.05.
- **depth**: 6.
- **iterations**: 200.
- **l2_leaf_reg**: 10 (giảm độ phức tạp).

#### TabNet
- **n_d, n_a**: 64 (chiều rộng các lớp quyết định và lớp chú ý).
- **n_steps**: 5 (số bước trong kiến trúc TabNet).
- **gamma**: 1.5.
- **lambda_sparse**: 1e-4 (giảm độ phức tạp).
- **optimizer_fn**: Adam.
- **scheduler_params**: ReduceLROnPlateau (điều chỉnh tốc độ học).

### 3.2. Tối ưu hóa ngưỡng
- **Mục đích**: Chuyển đổi giá trị liên tục thành phân loại (None, Mild, Moderate, Severe).
- **Phương pháp**:
  1. **Khởi tạo ngưỡng ban đầu**: [0.5, 1.5, 2.5].
  2. **Tối ưu hóa ngưỡng**: Sử dụng hàm `minimize` từ `scipy.optimize`.
  3. **Chuyển đổi dự đoán**: Áp dụng các ngưỡng tối ưu.
- **Kết quả**: QWK cải thiện đáng kể sau tối ưu hóa ngưỡng.

---

## 4. Kết luận
- Mô hình kết hợp nhiều thuật toán học máy (LightGBM, XGBoost, CatBoost, TabNet) thông qua Voting Regressor đạt hiệu suất cao.
- Kỹ thuật xử lý tính năng và điều chỉnh siêu tham số giúp tối ưu hóa mô hình.
- Kết quả cuối cùng được lưu trong `submission.csv`, sẵn sàng để nộp cho cuộc thi Kaggle.

