import argparse
import numpy as np
import random
import torch
import os
import gc
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_path = os.path.join(args.data_dir, "label.txt")
    wav_dir = os.path.join(args.data_dir, "wav")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"label.txt not found at {label_path}")

    print("Loading label.txt...")
    records = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 2: continue
            
            fname, lab = parts
            # Ensure the filename ends with .wav if it doesn't already
            if not fname.lower().endswith(".wav"):
                fname = fname + ".wav"
            
            records.append((fname, int(lab)))

    # Filter records to only those that actually exist on disk to prevent "System error"
    existing_records = [r for r in records if os.path.exists(os.path.join(wav_dir, r[0]))]
    print(f"Total samples in label.txt: {len(records)}")
    print(f"Samples found on disk: {len(existing_records)}")

    # Split by class
    bona = [r for r in existing_records if r[1] == 0]
    spoof = [r for r in existing_records if (r[1] == 1 or r[1] == 2)] # Adjust based on ADD2023 mapping

    n = min(args.n_each, len(bona), len(spoof))
    print(f"Selecting {n} samples from each class (Total: {2*n})")
    
    bona_sel = random.sample(bona, n)
    spoof_sel = random.sample(spoof, n)
    selected = bona_sel + spoof_sel

    print("Loading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    final_embeddings = []
    final_labels = []

    for i, (fname, lab) in enumerate(selected):
        file_path = os.path.join(wav_dir, fname)
        try:
            # Using librosa as a fallback as it is more robust than soundfile for weird headers
            wav, sr = librosa.load(file_path, sr=16000)

            inputs = processor(
                wav,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                out = model(inputs.input_values.to(device))
                # Mean pooling over time dimension
                emb = out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy()

            final_embeddings.append(emb)
            final_labels.append(0 if lab == 0 else 1)

        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(selected)} processed...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    X = np.vstack(final_embeddings)
    y = np.array(final_labels)

    np.savez_compressed(args.out_npz, X=X, y=y)
    print(f"Saved to {args.out_npz}. Final shape: {X.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Track3/train")
    parser.add_argument("--n_each", type=int, default=300)
    parser.add_argument("--out_npz", type=str, default="add_embeddings.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)