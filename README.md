##  Project: Layout Organization Recognition for Early Modern Printed Sources

This project is a part of the **RenAIssance GSoC 2025 evaluation**, specifically addressing **Test I: Layout Organization Recognition**. The goal is to develop a model that can accurately recognize and localize the **main textual regions** in early modern printed sources, while **ignoring decorative or non-informative elements** such as embellishments, decorations, and marginalia.The dataset comprises **6 scanned PDF sources**, each exhibiting unique Renaissance-era book layouts. Each PDF has been converted to images, and annotations have been made using bounding boxes to mark relevant layout regions like `main_text`, `heading`, `author`, and `drop_cap`.

##  Project Goals

| Objective                         | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
|  Layout Detection                 | Accurately detect `main_text`, `headings`, `drop_caps`, and `authors`.      |
|  Model Architecture               | Utilize YOLOv8 with ViT backbone for layout-aware object detection.         |
|  Evaluation Metrics               | Achieve high mAP@0.5 and mAP@0.5:0.95 using a clean, annotated dataset.     |
|  Historical Document Adaptation   | Focus on Renaissance-era inconsistencies: typography, layout, decorations.  |
|  Deployment-ready Codebase        | Structured for reproducible training, evaluation, and prediction.           |

### Why YOLO?

The YOLO (You Only Look Once) model was chosen for this project due to its following strengths:

1. **Real-time Performance:** YOLO's speed and efficiency make it suitable for processing large datasets of historical documents.
2. **Accuracy and Generalization:** YOLO models offer high accuracy in object detection and can generalize well across different document styles.
3. **Multi-Class Detection:** YOLO's architecture allows for simultaneous detection of multiple layout elements, simplifying the analysis process.
4. **Ease of Implementation and Training:** YOLOv8's user-friendly framework facilitates model development and fine-tuning.

These factors make YOLO a well-suited choice for the layout organization recognition task in the context of historical document analysis.

Model and Approach:

- Utilized a YOLOv8 object detection model with fine-tuning on a custom annotated dataset.
- Dataset included multiple layout classes such as main_text, heading, drop_cap, and author.
= Training was performed over 50 epochs, with careful monitoring of losses and evaluation metrics.
= Data was formatted in YOLOv8 and trained using augmentation strategies.


### Project Pipeline:

1. **Data Preparation:** Convert your PDF documents to images and annotate them using Roboflow.
2. **Configuration:** Update the `Config` class in the code with your dataset paths and hyperparameters.
3. **Training:** Run the `train_model()` function to train the YOLOv8 model.
4. **Evaluation:** Run the `evaluate_model()` function to assess the model's performance.
5. **Prediction:** Run the `predict_on_test_images()` function to apply the model to new images.

### About the Dataset:

- **6 scanned early modern PDFs** — Each source displays unique layouts, fonts, and styles.
- Converted into **image format** (JPEG) for training and inference.
- Annotated using **Roboflow** with four main layout classes:
  - `main_text`
  - `heading`
  - `drop_cap`
  - `author`
- Annotations are exported in **YOLOv8 format** (TXT files + data.yaml).
- Emphasis is on detecting **main reading content**, not decorative print elements.

### 📊 Evaluation Metrics (on Validation Set)

| **Metric**       | **Value** | **Description**                                                                                  |
|------------------|-----------|--------------------------------------------------------------------------------------------------|
| **mAP@0.5**       | 99.5%     | Mean Average Precision at IoU threshold 0.5 – measures how accurately the model predicts bounding boxes. |
| **mAP@0.5:0.95**  | 74.2%     | Mean Average Precision averaged across multiple IoU thresholds (0.5 to 0.95) – a stricter and more comprehensive performance metric. |
| **Precision**     | 92.5%     | Percentage of correctly predicted positive instances – high precision indicates few false positives. |
| **Recall**        | 98.6%     | Percentage of actual positives correctly predicted – high recall indicates few false negatives.   |
| **F1 Score**      | 95.5%     | Harmonic mean of Precision and Recall – provides a balanced accuracy measure.                   |

### Graphs

The project includes the following graphs for visualization:

- **Training and Validation Metrics (results.png):** Shows the trend of losses and evaluation metrics over epochs, providing insights into the model's learning process and potential overfitting.

![image](https://github.com/user-attachments/assets/bc6eae9c-07b9-443e-aaec-56a07b66dc01)

- **Precision and Recall Confidence Curve (P_curve.png, R_curve.png):** Illustrates the precision and recall of the model at different confidence thresholds, helping to understand the trade-off between these metrics.

![image](https://github.com/user-attachments/assets/275957bf-6846-4c62-bd6b-20a71c01e075)

- **mAP@0.5 and mAP@0.5:0.95 Trends:** Visualizes the improvement in detection accuracy over epochs, using both lenient and stricter IoU thresholds for evaluation.

![image](https://github.com/user-attachments/assets/a4c06843-5aca-401a-b1ba-4b1fbf6ae3f1)

-These graphs help to assess the model's performance, identify areas for improvement, and understand the impact of different hyperparameters.

## Predicted Images and Observations

After running inference with the trained YOLOv8 model, predicted images are generated with bounding boxes and class labels overlaid on the original document scans. These images showcase the model's ability to accurately detect and classify layout elements.

**Observations:**

- **Generalization:** The model demonstrates robustness across various fonts, layouts, and page structures commonly found in early modern printed sources.
- **Confidence:** High confidence scores associated with predictions indicate the model's strong predictive power and accurate feature learning.
- **Multi-Class Detection:** The model effectively detects multiple layout elements simultaneously, providing a holistic analysis of the document layout.
- **Clear Bounding Boxes:**  Bounding boxes are precise and non-overlapping, preserving the structural clarity of the document for downstream tasks.

**Example Images:**
![image](https://github.com/user-attachments/assets/44672a64-d74a-41e1-9d64-f9f08176423a)

### Results

- The model achieved high accuracy on the validation set (mAP@0.5: 0.9950, mAP@0.5:0.95: 0.7419).
- Detailed evaluation metrics and per-class results are available in the notebook.

### Next Steps

- Integrate with an OCR pipeline for text extraction.
- Experiment with advanced layout models (LayoutLMv3, Mask2Former + DINOv2).
- Enhance dataset quality and quantity.
- Post-processing and structuring output for downstream analysis.

###  Requirements  

- **Python 3.7+**
- **GPU (Recommended): A Tesla T4 (used in Colab) or any NVIDIA GPU with at least 8GB VRAM for faster training.**  
- **Ultralytics YOLOv8**  
- **Torch, Pandas, Matplotlib, Pillow (PIL), Fitz**

---

### Resources

- **Trained Model Weights**  
  [Download YOLOv8m Weights (Google Drive)](https://drive.google.com/drive/folders/1DerlCBIXqfETTMyN9ek10r3XGwFLL_rB)

- **Annotated Dataset**  
  [Download Annotated Dataset (Google Drive)](https://drive.google.com/drive/folders/11_eDbllbZIbj3x26ko0Im0lhujzUJpqt)

---

###  Installation 

#### 1️⃣ Clone the Repository  
```bash
https://github.com/Nikki370/Layout-Organization-Recognition-for-early-modern-printed-source
```
#### 2️⃣ Install Required Packages
```bash
!pip install pandas
!pip install matplotlib
!pip install pymupdf
!pip install pillow
!pip install ultralytics
!pip install torch
```
---

### 👨‍💻 Author  
📌 **Nikita Kumari**  
📧 [np810652@gmail.com] | 🖥️ [https://github.com/Nikki370] (link to github)  

🔹 If you find this project useful, give it a ⭐ on GitHub! 🚀  



---

## 🔐 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project with proper credit.

See the [LICENSE](./LICENSE) file for full details.



