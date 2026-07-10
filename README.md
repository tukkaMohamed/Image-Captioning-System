# 🖼️ Image Captioning System — Streamlit GUI

Multimodal AI demo comparing **LSTM**, **GRU**, and **Transformer** decoders on image captioning.

## 📁 Project Structure

```
ImageCaptioningGUI/
│
├── app.py               ← Streamlit entry point  →  streamlit run app.py
├── predict.py           ← High-level predict(image, model) → caption
├── model_loader.py      ← Loads all 3 models (unified or legacy checkpoints)
├── requirements.txt
├── README.md
│
├── models/
│   ├── LSTM/
│   │   ├── lstm_full.pth          ← unified checkpoint  (model+vocab+config)
│   │   │   OR
│   │   ├── lstm_attn.pth          ← legacy weights
│   │   └── vocab.pkl              ← legacy vocab
│   │
│   ├── GRU/
│   │   ├── gru_full.pth           ← unified checkpoint
│   │   │   OR
│   │   ├── gru_caption_model.pth  ← legacy weights
│   │   └── vocab.pkl              ← legacy vocab
│   │
│   └── Transformer/
│       └── transformer_best.pth   ← unified checkpoint (always unified)
│
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py  ← EfficientNetB0 → (49, 1280)
│   └── caption_utils.py      ← greedy decode (LSTM/GRU) + beam search (Transformer)
│
├── assets/
│   └── logo.png               ← optional logo
│
└── uploaded_images/           ← auto-created, stores last uploaded image
```

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Checkpoint Formats

### Unified (recommended — no mismatch errors ever)
```python
torch.save({
    "model_state": model.state_dict(),
    "vocab":       vocab,
    "vocab_size":  len(vocab),
    "config":      {"embed_dim": 256, "hidden_dim": 512, ...}
}, "models/LSTM/lstm_full.pth")
```

### Legacy (separate files — still supported)
- `lstm_attn.pth` + `vocab.pkl`  in `models/LSTM/`
- `gru_caption_model.pth` + `vocab.pkl` in `models/GRU/`

The app auto-detects which format is present.

## 🎮 Features

| Feature | Description |
|---|---|
| Single model | Upload image → pick LSTM / GRU / Transformer → get caption |
| Compare All | Runs all 3 models simultaneously, shows captions + inference times |
| Project Info | Dataset stats, architecture overview, performance table, pipeline diagram |
