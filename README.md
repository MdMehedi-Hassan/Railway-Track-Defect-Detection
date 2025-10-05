# RailTrack Inspector: An Explainable Deep Learning–Driven Web Application to Detect and Localize Railway Track Defects**

---

## 📖 Overview

**Railways are an essential part of international transportation, but the majority of defects detection methods rely on manual inspection, which is inefficient, error-prone, and increasingly inadequate for ensuring safety under growing rail traffic. Structural anomalies such as missing or broken fasteners, missing bolts, damaged fishplates, and rail cracks frequently remain undetected through manual inspection, posing risks of severe accidents and economic loss. To address this challenge, we proposed a Deep Learning-based Web Application to Detect Rail Track Defects with Location. We deployed our proposed model Faster R-CNN in a lightweight web application supporting real-time inference, geolocation-based defect mapping, and interpretability via Grad-CAM visualizations. This integration not only enables automated detection and precise localization of defects but also provides transparent decision support to railway maintenance teams. While validated on Bangladesh’s railway infrastructure, the framework is scalable and adaptable to global railway networks.**

---

## 🧠 Key Features

- 🚄 **Automated Detection:** Detects multiple railway track defects from captured images.
- 🌍 **Location Mapping:** Integrates GPS metadata for precise localization of defects.
- 🧩 **Explainable AI:** Grad-CAM visualization highlights the regions influencing predictions.
- 💻 **Web-Based Interface:** Built using **Python Streamlit** for ease of deployment and real-time monitoring.
- 📦 **Database Integration:** All defect reports and GPS data are stored in **MongoDB**.
- ⚙️ **Model Comparison:** Performance comparison between Faster R-CNN and YOLO11.
---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Framework** | Python 3.10+, Streamlit |
| **Deep Learning** | PyTorch, TensorFlow |
| **Model** | Faster R-CNN (ResNet-50 backbone), YOLO11 |
| **Database** | MongoDB |
| **Visualization** | Grad-CAM (Explainable AI) |
| **Annotation Tool** | CVAT.ai |
| **Frontend** | Streamlit UI |
| **Backend** | Flask API or integrated Streamlit backend |

---

## 🗂️ Dataset Description

### Primary Dataset
- **Total Images:** 1,200  
- **Locations:**  
  - Feni Railway Station – 669 images  
  - Chittagong Railway Station – 531 images  
- **Classes:**
  - 469 Defect images  
  - 731 Non-Defect images  

### Secondary Dataset
- **Total Images:** 1,250 (750 Defect, 500 Non-Defect)  
- **Subclasses:**
  - Missing Bolts  
  - Missing Fasteners  
  - Rail Cracks  

> Data annotated using **CVAT.ai** and augmented with rotation, flipping, brightness adjustment, and blurring to improve generalization.

---

## 🧮 Model Details

### 1️⃣ Faster R-CNN
- Backbone: **ResNet-50**
- Region Proposal Network (RPN) for candidate object generation
- Achieved **98% accuracy**, **0.9849 F1-score**, and **0.9679 mAP@0.5**

### 2️⃣ YOLO11
- Faster inference, high recall capability
- Achieved **97.5% accuracy**, **0.972 F1-score**, and **0.985 mAP@0.5**

| Metric | Faster R-CNN | YOLO11 |
|--------|---------------|--------|
| Accuracy | 0.98 | 0.975 |
| F1-Score | 0.9849 | 0.972 |
| Recall | 0.98 | 0.969 |
| Precision | 0.9899 | 0.975 |
| mAP@0.5 | 0.9679 | 0.985 |

---

## 🧾 Evaluation Metrics

- **Accuracy** – Overall correct predictions  
- **Precision** – Fraction of true positives among predicted positives  
- **Recall** – Fraction of true positives among actual positives  
- **F1-Score** – Harmonic mean of precision and recall  
- **mAP@0.5** – Mean Average Precision across all classes  

---

## 🔍 Explainability

Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** provides heatmaps showing the regions influencing the model’s decision.  
- 🔴 Red: High importance  
- 🔵 Blue: Low importanc  
This enhances transparency and reliability for railway safety applications.

---

## 🌐 Web Application

The deployed **Streamlit Web App** provides:
- 📸 Image upload interface  
- 🧩 Real-time defect detection using the trained model  
- 📍 Display of defect coordinates on an embedded map  
- 🧾 Automated defect report generation  
- 💾 Storage in MongoDB for maintenance tracking  

```bash
# Run the app locally
streamlit run app.py
