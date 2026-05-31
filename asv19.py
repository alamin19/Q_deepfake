import argparse, time, torch, gc, numpy as np, pandas as pd, matplotlib.pyplot as plt
import os, glob
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# ==========================================
# 1. METRICS & UTILS
# ==========================================
def compute_metrics(y_true, y_score, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = (fpr[np.argmin(np.abs(fpr - fnr))] + fnr[np.argmin(np.abs(fpr - fnr))]) / 2

    ece, bin_boundaries = 0.0, np.linspace(0, 1, 11)
    for i in range(10):
        mask = (y_score > bin_boundaries[i]) & (y_score <= bin_boundaries[i+1])
        if np.any(mask):
            ece += np.abs(np.mean(y_score[mask]) - np.mean(y_true[mask])) * np.mean(mask)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred), "f1": f1, "auc": roc_auc_score(y_true, y_score), "eer": eer, "ece": ece}


# ==========================================
# 2. DATA ENGINE (HUGGING FACE DATASET)
# ==========================================
def extract_hf_data(n_each):
    print("Loading ASVspoof_2019_LA dataset from Hugging Face...")
    # Load the train split
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train")
    
    # Convert to pandas to easily perform balanced sampling by 'key'
    df = pd.DataFrame({
        'index': range(len(ds)),
        'key': ds['key']
    })
    
    # 0 = Bonafide, 1 = Spoof based on your configuration
    spoof_df = df[df["key"] == 1].sample(n_each, random_state=42)
    bonafide_df = df[df["key"] == 0].sample(n_each, random_state=42)
    selected_indices = pd.concat([spoof_df, bonafide_df])['index'].tolist()
    
    print(f"Sampled {len(selected_indices)} total audios ({n_each} spoof, {n_each} bonafide).")
    
    # Load feature extractor models
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    audios = []
    labels = []
    
    # Extract features for sampled indices
    for idx in selected_indices:
        example = ds[int(idx)]
        
        # Hugging Face Audio feature returns a dict: {'array': np.array, 'sampling_rate': int}
        speech = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        
        # Resample if dataset rate isn't natively 16kHz
        if sr != 16000:
            import librosa
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        inputs = proc(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            features = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
            
        audios.append(features)
        # Assign target label based on the 'key' rule provided
        labels.append(0 if example["key"] == 0 else 1)
        
    return np.array(audios), np.array(labels)


# ==========================================
# 3. MAIN PROCESS
# ==========================================
def main(args):
    # Data Engine
    X, y = extract_hf_data(args.n_each)
    X_pca = PCA(n_components=args.qubits).fit_transform(StandardScaler().fit_transform(X))
    X_q = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_pca)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = ["Classical_SVM", "Classical_MLP", "Quantum_SVM"]
    stats = {m: [] for m in models}
    scores_log = {m: [] for m in models}
    y_log = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_pca, y), 1):
        print(f"--- Fold {fold} ---")
        y_te = y[te_idx]
        y_log.append(y_te)

        # Define Classifiers
        clfs = {
            "Classical_SVM": SVC(kernel="rbf", probability=True, C=args.c_val),
            "Classical_MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
            "Quantum_SVM": QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=ZZFeatureMap(args.qubits, reps=1, entanglement='linear')))
        }

        for name, obj in clfs.items():
            xtr = X_q[tr_idx] if "Quantum" in name else X_pca[tr_idx]
            xte = X_q[te_idx] if "Quantum" in name else X_pca[te_idx]

            obj.fit(xtr, y[tr_idx])
            prob = obj.decision_function(xte) if "Quantum" in name else obj.predict_proba(xte)[:, 1]
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-10) # Norm

            stats[name].append(compute_metrics(y_te, prob, obj.predict(xte)))
            scores_log[name].append(prob)
            
    print("\n" + "="*60 + "\nRESULTS SUMMARY\n" + "="*60)
    for m in models:
        avg = {k: np.mean([x[k] for x in stats[m]]) for k in ["acc", "f1", "auc", "eer", "ece"]}
        print(f"{m:<15} | Acc: {avg['acc']:.4f} | EER: {avg['eer']:.4f} | ECE: {avg['ece']:.4f}")

    # Simplified Plotter
    plt.figure(figsize=(10, 5))
    for m in models:
        fpr, tpr, _ = roc_curve(np.concatenate(y_log), np.concatenate(scores_log[m]))
        plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(np.concatenate(y_log), np.concatenate(scores_log[m])):.2f})")
    plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("ROC Comparison"); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_each", type=int, default=50) # Samples per class
    parser.add_argument("--qubits", type=int, default=2)  
    parser.add_argument("--c_val", type=float, default=10.0)
    main(parser.parse_args())