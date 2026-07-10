"""
utils/caption_utils.py
Caption generation for:
- LSTM (Bahdanau Attention)
- GRU (Bahdanau Attention)
- Transformer (Beam Search)
"""

import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Greedy decoding (LSTM / GRU)
# ==========================================================

@torch.no_grad()
def generate_greedy(
    model,
    img_feat,
    vocab,
    idx2word,
    max_len=25
):

    model.eval()

    feat = torch.tensor(
        img_feat,
        dtype=torch.float32,
        device=DEVICE
    )

    if feat.dim() == 2:
        feat = feat.unsqueeze(0)      # (1,49,1280)

    is_lstm = hasattr(model, "lstm")

    h = torch.zeros(
        1,
        model.decoder_dim,
        device=DEVICE
    )

    if is_lstm:
        c = torch.zeros_like(h)

    word = torch.tensor(
        [[vocab["<start>"]]],
        device=DEVICE
    )

    caption = []

    for _ in range(max_len):

        emb = model.embedding(word).squeeze(1)

        context, _ = model.attention(
            feat,
            h
        )

        decoder_input = torch.cat(
            (
                emb,
                context
            ),
            dim=1
        )

        if is_lstm:

            h, c = model.lstm(
                decoder_input,
                (h, c)
            )

        else:

            h = model.gru(
                decoder_input,
                h
            )

        logits = model.fc(
            model.dropout(h)
        )

        pred = logits.argmax(1).item()

        if pred == vocab["<end>"]:
            break

        caption.append(
            idx2word.get(
                pred,
                "<unk>"
            )
        )

        word = torch.tensor(
            [[pred]],
            device=DEVICE
        )

    return " ".join(caption)


# ==========================================================
# Beam Search (Transformer)
# ==========================================================

@torch.no_grad()
def generate_beam(
    model,
    img_feat,
    vocab,
    idx2word,
    beam_width=3,
    max_len=30
):

    model.eval()

    img_feat = torch.tensor(
        img_feat,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    memory = model.img_proj(img_feat)

    beams = [

        (
            0.0,
            [vocab["<start>"]]
        )

    ]

    completed = []

    for _ in range(max_len):

        new_beams = []

        for score, tokens in beams:

            if tokens[-1] == vocab["<end>"]:

                completed.append(
                    (
                        score,
                        tokens
                    )
                )

                continue

            tgt = torch.tensor(
                [tokens],
                device=DEVICE
            )

            tgt_emb = model.pos_enc(

                model.embedding(tgt)
                * math.sqrt(model.d_model)

            )

            mask = model.make_causal_mask(
                tgt.size(1),
                DEVICE
            )

            out = model.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=mask
            )

            logits = model.fc_out(
                out[:, -1, :]
            )

            log_probs = torch.log_softmax(
                logits,
                dim=-1
            )[0]

            values, indices = log_probs.topk(
                beam_width
            )

            for lp, idx in zip(
                values.tolist(),
                indices.tolist()
            ):

                new_beams.append(

                    (
                        score + lp,
                        tokens + [idx]
                    )

                )

        new_beams.sort(
            key=lambda x: x[0],
            reverse=True
        )

        beams = new_beams[:beam_width]

        if len(completed) >= beam_width:
            break

    if not completed:
        completed = beams

    completed.sort(
        key=lambda x: x[0] / len(x[1]),
        reverse=True
    )

    best = completed[0][1]

    words = [

        idx2word.get(t, "<unk>")

        for t in best

        if t not in (

            vocab["<start>"],
            vocab["<end>"],
            vocab["<pad>"]

        )

    ]

    return " ".join(words)