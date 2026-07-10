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

<img width="1918" height="1078" alt="Screenshot 2026-07-10 220539" src="https://github.com/user-attachments/assets/df524170-b1f3-4951-835a-3f7531e2731f" />

-----
<img width="1918" height="1078" alt="Screenshot 2026-07-10 221012" src="https://github.com/user-attachments/assets/6d3c0428-b642-46e2-b4f7-d79bdee6a5d6" />

-----
<img width="1918" height="1078" alt="Screenshot 2026-07-10 220952" src="https://github.com/user-attachments/assets/75992106-7d6c-4814-be7e-42cf49e81c42" />

-----
<img width="1918" height="875" alt="Screenshot 2026-07-10 220811" src="https://github.com/user-attachments/assets/646752c8-1e4b-4ba1-8b73-93a784c2d97a" />

-----

<img width="1918" height="1078" alt="Screenshot 2026-07-10 220558" src="https://github.com/user-attachments/assets/03e5e4e1-ea87-4ace-bc0f-17914a93df17" />

-----
<img width="1918" height="1078" alt="Screenshot 2026-07-10 220645" src="https://github.com/user-attachments/assets/6fa329ac-444e-4171-b7c6-368dad176173" />

----

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
