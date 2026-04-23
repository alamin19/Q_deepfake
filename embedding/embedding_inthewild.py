import argparse
import numpy as np
import random
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import gc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading In The Wild dataset...")
    ds = load_dataset("UncovAI/InTheWild", split="train")

    # Cast audio column to 16kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Convert string label
    def label_to_int(example):
        #print(example["key"]) 
        example["label"] = 0 if example["label"][0] == 0 else 1
        return example

    ds = ds.map(label_to_int)

    # Instead of iterating through entire dataset, work with indices
    print("Splitting dataset by label...")
    bonafide_indices = [i for i, x in enumerate(ds) if x["label"] == 0]
    spoof_indices = [i for i, x in enumerate(ds) if x["label"] == 1]

    print(f"Bonafide: {len(bonafide_indices)}, Spoof: {len(spoof_indices)}")

    n = args.n_each

    # Sample indices instead of full examples
    bonafide_sel_idx = random.sample(bonafide_indices, n)
    spoof_sel_idx = random.sample(spoof_indices, n)

    selected_indices = bonafide_sel_idx + spoof_sel_idx
    labels = [0] * n + [1] * n

    print("Loading Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()

    embeddings = []

    print(f"Extracting embeddings for {2*n} samples...")
    for i, idx in enumerate(selected_indices):
        try:
            # Access one item at a time to reduce memory pressure
            item = ds[idx]
            wav = torch.tensor(item["audio"]["array"]).float()

            inputs = processor(
                wav.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                out = model(inputs.input_values.to(device))
                last = out.last_hidden_state.squeeze(0)
                emb = last.mean(dim=0).cpu().numpy()

            embeddings.append(emb)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{2*n} extracted")
                # Periodic garbage collection to free resources
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Use zero embedding as placeholder for failed samples
            embeddings.append(np.zeros(768))  # wav2vec2-base has 768 dims
            continue

    print(f"Successfully extracted {len(embeddings)}/{2*n} embeddings")

    X = np.vstack(embeddings)
    y = np.array(labels)

    np.savez_compressed(args.out_npz, X=X, y=y)
    print("Saved:", args.out_npz)
    print("Shape:", X.shape, "Labels:", y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_each", type=int, default=300)
    parser.add_argument("--out_npz", type=str, default="iwild_embeddings.npz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)