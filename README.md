# 🖼️ AI Image Captioning System

An AI-powered Image Captioning System built with **PyTorch** and **Streamlit** that automatically generates natural language descriptions for images. The project compares three deep learning architectures (**LSTM**, **GRU**, and **Transformer**) using **EfficientNetB0** as the image feature extractor.

---

## 🚀 Features

- Upload an image and generate an automatic caption.
- Compare three different captioning models:
  - LSTM + Bahdanau Attention
  - GRU + Bahdanau Attention
  - Transformer Decoder
- Interactive Streamlit web interface.
- Compare inference time between models.
- Modern and user-friendly UI.

---

## 🧠 Models

| Model | Architecture | Decoding |
|--------|--------------|----------|
| LSTM | Bahdanau Attention + LSTMCell | Greedy Search |
| GRU | Bahdanau Attention + GRUCell | Greedy Search |
| Transformer | Transformer Decoder | Beam Search |

---

## 📊 Dataset

- Flickr8k Dataset
- Image Encoder: EfficientNetB0
- Extracted Feature Size: **49 × 1280**

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Streamlit
- TorchVision
- EfficientNetB0
- PIL (Pillow)
- NumPy

---

## 📁 Project Structure

```text
ImageCaptioningGUI/
│
├── app.py
├── predict.py
├── model_loader.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── LSTM/
│   ├── GRU/
│   └── Transformer/
│
├── utils/
│   ├── feature_extractor.py
│   └── caption_utils.py
│
└── uploaded_images/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/tukkaMohamed/Image-Captioning-System.git
```

Go to the project directory

```bash
cd Image-Captioning-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🔄 Project Pipeline

```text
Input Image
      │
      ▼
EfficientNetB0
      │
Extract Image Features
      │
      ▼
LSTM / GRU / Transformer
      │
      ▼
Generated Caption
```

---

## 📸 Screenshots

### Home Page

> Add a screenshot here.

```
screenshots/home.png
```

### Caption Generation

> Add a screenshot here.

```
screenshots/caption.png
```

### Compare Models

> Add a screenshot here.

```
screenshots/compare.png
```

---

## 📈 Results

The application allows comparing:

- Generated captions from all three models.
- Inference time for each model.
- Side-by-side qualitative comparison.

---

## 🎯 Future Improvements

- Deploy the application online.
- Add BLEU, METEOR, and ROUGE evaluation inside the interface.
- Train on the MS COCO dataset.
- Add attention visualization.
- Support custom model checkpoints.

---
