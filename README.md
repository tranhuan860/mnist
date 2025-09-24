# 🧠 Mô hình CNN phân loại chữ số viết tay trên tập dữ liệu MNIST

## 📂 Tập dữ liệu
- **MNIST** gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra.
- Mỗi ảnh kích thước **28x28 pixel**, ảnh grayscale.

## 🎯 Kết quả tổng quan
- **Accuracy**: 0.9961  
- **Macro Precision**: 0.9961  
- **Macro Recall**: 0.9961  
- **Macro F1-score**: 0.9961  

## 📊 Kết quả theo từng lớp

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

## 📝 Báo cáo phân loại chi tiết

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

📌 **Ghi chú:**  
- **Macro Avg** = trung bình cộng các lớp.  
- **Weighted Avg** = trung bình có trọng số theo support từng lớp. 

## 📈 Hình ảnh minh họa

- **Biểu đồ loss:**

![Loss](image/loss.png)

- **Biểu đồ accuracy**

![Accuracy](image/accuracy.png)

- **Biểu đồ nhầm lẫn (Confusion Matrix):**

![Confusion Matrix](image/confusion_matrix_0.png)

- **Biểu đồ ROC Curve:**

![ROC Curve](image/roc_curve.png)

---

✨ *Mô hình đạt độ chính xác cao nhờ kiến trúc CNN tối ưu, huấn luyện trên MNIST với các kỹ thuật regularization và augmentation hợp lý.*
