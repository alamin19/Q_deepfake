import argparse
import random
import gc
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# --------------------------------------------------
# Load protocol → filename → label
# --------------------------------------------------
def load_protocol(tsv_path):
    mapping = {}
    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            fname = parts[1]          # T_0000000000
            label_str = parts[-2]     # spoof / bonafide

            if label_str == "bonafide":
                mapping[fname] = 0
            elif label_str == "spoof":
                mapping[fname] = 1

    return mapping


# --------------------------------------------------
# Load FLAC safely
# --------------------------------------------------
def load_flac(path):
    wav, sr = torchaudio.load(path)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    if wav.numel() == 0:
        raise ValueError("Empty audio")

    return wav.squeeze(0)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading protocol...")
    proto = load_protocol(args.protocol)
    print(f"Protocol entries: {len(proto)}")

    print("Scanning audio directory (.flac)...")
    audio_files = list(Path(args.audio_dir).rglob("*.flac"))
    print(f"Found audio files: {len(audio_files)}")

    # --------------------------------------------------
    # Build VALID pool: audio ∩ protocol
    # --------------------------------------------------
    valid = []

    for p in audio_files:
        fname = p.stem  # T_0000000000

        if fname not in proto:
            continue

        valid.append((p, proto[fname]))

    print(f"Valid audio+label pairs: {len(valid)}")

    # Split by class
    bonafide = [x for x in valid if x[1] == 0]
    spoof = [x for x in valid if x[1] == 1]

    print(f"Bonafide valid: {len(bonafide)}")
    print(f"Spoof valid:    {len(spoof)}")

    if len(bonafide) < args.n_each or len(spoof) < args.n_each:
        raise RuntimeError("Not enough valid samples to draw from")

    # --------------------------------------------------
    # Sample AFTER validation
    # --------------------------------------------------
    bonafide_sel = random.sample(bonafide, args.n_each)
    spoof_sel = random.sample(spoof, args.n_each)

    selected = bonafide_sel + spoof_sel
    labels = [0] * args.n_each + [1] * args.n_each

    print(f"Selected total: {len(selected)}")

    # --------------------------------------------------
    # Load Wav2Vec2
    # --------------------------------------------------
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    ).to(device)
    model.eval()

    embeddings = []
    failures = 0

    print("Extracting embeddings...")

    for i, (path, label) in enumerate(selected):
        try:
            wav = load_flac(path)

            inputs = processor(
                wav.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                out = model(inputs.input_values.to(device))
                last = out.last_hidden_state.squeeze(0)

                if torch.isnan(last).any():
                    raise ValueError("NaNs detected")

                emb = last.mean(dim=0).cpu().numpy()

            embeddings.append(emb)

            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(selected)} done")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            failures += 1
            print(f"[FAILED] {path.name}: {e}")

    print(f"Failures during extraction: {failures}")

    X = np.vstack(embeddings)
    y = np.array(labels[:len(X)])

    np.savez_compressed(args.out_npz, X=X, y=y)

    print("Saved:", args.out_npz)
    print("Final shape:", X.shape, y.shape)


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/content/flac_T_ab/flac_T"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="/content/ASVspoof5_protocols/train.tsv"
    )
    parser.add_argument("--n_each", type=int, default=300)
    parser.add_argument("--out_npz", type=str, default="asvspoof5_flac_embeddings.npz")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
