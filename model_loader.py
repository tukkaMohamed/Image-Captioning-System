
import math
import os
import pickle

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Bahdanau Attention  (exact copy from training notebooks)
# ─────────────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att    = nn.Linear(attention_dim, 1)
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1   = self.encoder_att(encoder_out)
        att2   = self.decoder_att(decoder_hidden).unsqueeze(1)
        energy = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha  = self.softmax(energy)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


# ─────────────────────────────────────────────────────────────
# AttentionLSTM  (exact copy from 03_LSTM_Training.ipynb)
# ─────────────────────────────────────────────────────────────
class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, encoder_dim=1280, embed_dim=256,
                 decoder_dim=512, attention_dim=512, dropout=0.5):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention   = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm        = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(decoder_dim, vocab_size)

    def forward(self, features, captions):
        batch_size  = features.size(0)
        embeddings  = self.embedding(captions)
        h = torch.zeros(batch_size, self.decoder_dim, device=features.device)
        c = torch.zeros_like(h)
        outputs = []
        for t in range(captions.size(1) - 1):
            context, _ = self.attention(features, h)
            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1)
            h, c       = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(self.dropout(h)))
        return torch.stack(outputs, dim=1)


# ─────────────────────────────────────────────────────────────
# AttentionGRU  (exact copy from 04_GRU_Training.ipynb)
# ─────────────────────────────────────────────────────────────
class AttentionGRU(nn.Module):
    def __init__(self, vocab_size, encoder_dim=1280, embed_dim=256,
                 decoder_dim=512, attention_dim=512, dropout=0.5):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention   = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.gru         = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(decoder_dim, vocab_size)

    def forward(self, features, captions):
        batch_size  = features.size(0)
        embeddings  = self.embedding(captions)
        h = torch.zeros(batch_size, self.decoder_dim, device=features.device)
        outputs = []
        for t in range(captions.size(1) - 1):
            context, _ = self.attention(features, h)
            gru_input  = torch.cat((embeddings[:, t, :], context), dim=1)
            h          = self.gru(gru_input, h)
            outputs.append(self.fc(self.dropout(h)))
        return torch.stack(outputs, dim=1)


# ─────────────────────────────────────────────────────────────
# Transformer  (same as before — unified checkpoint)
# ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ImageCaptionTransformer(nn.Module):
    def __init__(self, vocab_size, img_dim=1280, d_model=256, nhead=8,
                 num_layers=3, dim_ff=1024, dropout=0.3, max_len=128):
        super().__init__()
        self.d_model   = d_model
        self.img_proj  = nn.Linear(img_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc_out  = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, img, tgt):
        memory   = self.img_proj(img)
        tgt_emb  = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))
        mask     = self.make_causal_mask(tgt.size(1), tgt.device)
        pad_mask = (tgt == 0)
        out      = self.decoder(tgt_emb, memory, tgt_mask=mask,
                                tgt_key_padding_mask=pad_mask)
        return self.fc_out(out)


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
LEGACY = {
    "LSTM": {
        "weights": os.path.join("models", "LSTM", "lstm_attn.pth"),
        "vocab":   os.path.join("models", "LSTM", "vocab.pkl"),
    },
    "GRU": {
        "weights": os.path.join("models", "GRU", "gru_caption_model.pth"),
        "vocab":   os.path.join("models", "GRU", "vocab.pkl"),
    },
}
TRANSFORMER_PATH = os.path.join("models", "Transformer", "transformer_best.pth")

DEFAULT_CFG = dict(encoder_dim=1280, embed_dim=256,
                   decoder_dim=512, attention_dim=512, dropout=0.5)


# ─────────────────────────────────────────────────────────────
# Public loader
# ─────────────────────────────────────────────────────────────
def load_model(name: str):
    """Returns (model, vocab, idx2word)."""

    if name in ("LSTM", "GRU"):
        lg    = LEGACY[name]
        cls   = AttentionLSTM if name == "LSTM" else AttentionGRU

        with open(lg["vocab"], "rb") as f:
            vocab = pickle.load(f)

        state      = torch.load(lg["weights"], map_location=DEVICE)
        vocab_size = state["embedding.weight"].shape[0]

        if vocab_size != len(vocab):
            raise RuntimeError(
                f"{name}: checkpoint vocab_size={vocab_size} "
                f"but vocab.pkl has {len(vocab)} words. "
                "They must come from the same training run."
            )

        model = cls(vocab_size=vocab_size, **DEFAULT_CFG).to(DEVICE)
        model.load_state_dict(state)
        model.eval()

    elif name == "Transformer":
        ckpt  = torch.load(TRANSFORMER_PATH, map_location=DEVICE)
        vocab = ckpt["vocab"]
        cfg   = {**{"img_dim":1280,"d_model":256,"nhead":8,
                    "num_layers":3,"dim_ff":1024,"dropout":0.3},
                 **ckpt.get("config", {})}
        model = ImageCaptionTransformer(vocab_size=ckpt["vocab_size"], **cfg).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

    else:
        raise ValueError(f"Unknown model: {name}")

    idx2word = {v: k for k, v in vocab.items()}
    return model, vocab, idx2word
