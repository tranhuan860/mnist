# 🧠 MÔ HÌNH CNN PHÂN LOẠI CHỮ SỐ VIẾT TAY – MNIST  

> Mô hình Convolutional Neural Network (CNN) huấn luyện trên tập dữ liệu **MNIST** để phân loại chữ số viết tay (0–9).  
> README này trình bày chi tiết kết quả, báo cáo phân loại và các hình ảnh minh họa trực quan.  

---

## 📂 1. Tập dữ liệu  

- **Nguồn**: MNIST (Yann LeCun et al.).  
- **Kích thước**:  
  - 60.000 ảnh huấn luyện.  
  - 10.000 ảnh kiểm tra.  
- **Đặc điểm ảnh**:  
  - Grayscale.  
  - Kích thước **28×28 pixel**.  

---

## 🎯 2. Kết quả tổng quan  

| Metric           | Giá trị |
|-----------------|----------|
| **Accuracy**    | 0.9961   |
| **Macro Precision** | 0.9961   |
| **Macro Recall**    | 0.9961   |
| **Macro F1-score**  | 0.9961   |

---

## 📊 3. Kết quả theo từng lớp  

| Lớp | Precision | Recall | F1-score |
|-----|-----------|--------|----------|
| 0   | 0.9980    | 0.9990 | 0.9985   |
| 1   | 0.9974    | 0.9974 | 0.9974   |
| 2   | 0.9961    | 0.9961 | 0.9961   |
| 3   | 0.9941    | 0.9970 | 0.9956   |
| 4   | 0.9939    | 0.9959 | 0.9949   |
| 5   | 0.9978    | 0.9944 | 0.9961   |
| 6   | 0.9958    | 0.9969 | 0.9963   |
| 7   | 0.9942    | 0.9942 | 0.9942   |
| 8   | 0.9979    | 0.9969 | 0.9974   |
| 9   | 0.9960    | 0.9931 | 0.9945   |

---

## 📝 4. Báo cáo phân loại chi tiết  

| Lớp | Precision | Recall | F1-score | Support |
|-----|-----------|--------|----------|---------|
| 0   | 1.00      | 1.00   | 1.00     | 980     |
| 1   | 1.00      | 1.00   | 1.00     | 1135    |
| 2   | 1.00      | 1.00   | 1.00     | 1032    |
| 3   | 0.99      | 1.00   | 1.00     | 1010    |
| 4   | 0.99      | 1.00   | 0.99     | 982     |
| 5   | 1.00      | 0.99   | 1.00     | 892     |
| 6   | 1.00      | 1.00   | 1.00     | 958     |
| 7   | 0.99      | 0.99   | 0.99     | 1028    |
| 8   | 1.00      | 1.00   | 1.00     | 974     |
| 9   | 1.00      | 0.99   | 0.99     | 1009    |

| Tổng hợp       | Accuracy | Macro Avg (P/R/F1) | Weighted Avg (P/R/F1) |
|----------------|----------|--------------------|-----------------------|
| **Toàn bộ**    | 1.00     | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00    |

---

📌 **Chú thích:**  
- **Macro Avg** = Trung bình cộng kết quả của tất cả các lớp.  
- **Weighted Avg** = Trung bình có trọng số theo số lượng mẫu mỗi lớp.  

---

## 📈 5. Hình ảnh minh họa  

### 🔹 Quá trình huấn luyện  

- **Biểu đồ Loss:**

  ![Loss](image/loss.png)

- **Biểu đồ Accuracy:**

  ![Accuracy](image/accuracy.png)

### 🔹 Đánh giá mô hình  

- **Ma trận nhầm lẫn (Confusion Matrix):**

  ![Confusion Matrix](image/confusion_matrix_0.png)

- **Đường cong ROC (ROC Curve):**

  ![ROC Curve](image/roc_curve.png)

---

## ✨ 6. Nhận xét  

> Mô hình CNN đạt **độ chính xác cao** nhờ:
> - Kiến trúc tối ưu cho nhận dạng ảnh số viết tay.
> - Sử dụng **regularization** và **data augmentation** hợp lý.
> - Quy trình huấn luyện và đánh giá nhất quán trên tập MNIST.

---

