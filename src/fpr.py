import argparse
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, det_curve
)
import matplotlib.pyplot as plt

# Qiskit imports for QSVM
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# ==========================================
# METRIC UTILITIES
# ==========================================

def compute_tpr_at_fpr(y_true, y_score, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return np.interp(target_fpr, fpr, tpr)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diff)
    return (fpr[min_index] + fnr[min_index]) / 2

def eval_metrics(y_true, y_score, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    eer = compute_eer(y_true, y_score)
    tpr_at_5 = compute_tpr_at_fpr(y_true, y_score, target_fpr=0.05)
    return {"acc": acc, "f1": f1, "auc": auc, "eer": eer, "tpr@5": tpr_at_5}

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_evaluation(train_path, test_path, args, log_file):
    n_runs = 3
    test_name = os.path.basename(test_path)
    
    # Models to evaluate
    model_list = ["SVM", "MLP"]
    if args.use_qsvm:
        model_list.insert(0, "QSVM")
    
    # Storage for statistics across 3 runs
    stats = {m: {met: [] for met in ["acc", "f1", "auc", "eer", "tpr@5"]} for m in model_list}

    print(f"\n>>> EVALUATING TEST CORPUS: {test_name}")
    
    for run in range(n_runs):
        # 1. Load Data
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        X_train_raw, y_train_raw = train_data["X"], train_data["y"]
        X_test_raw, y_test_raw = test_data["X"], test_data["y"]

        # 2. Subsample to 200 (changing seed per run for std deviation)
        def limit_data(X, y, size, seed):
            if len(X) > size:
                from sklearn.model_selection import train_test_split
                X, _, y, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
            return X, y

        X_train_sub, y_train = limit_data(X_train_raw, y_train_raw, 200, seed=42+run)
        X_test_sub, y_test = limit_data(X_test_raw, y_test_raw, 200, seed=100+run)

        # 3. Preprocessing (Fit on Train, Transform Test)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sub)
        X_test_s = scaler.transform(X_test_sub)

        pca = PCA(args.shared_dim)
        X_train = pca.fit_transform(X_train_s)
        X_test = pca.transform(X_test_s)

        # --- QSVM ---
        if args.use_qsvm:
            q_scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_tr_q = q_scaler.fit_transform(X_train)
            X_te_q = q_scaler.transform(X_test)
            fm = ZFeatureMap(feature_dimension=args.shared_dim, reps=args.qsvm_reps)
            q_ker = FidelityQuantumKernel(fidelity=ComputeUncompute(sampler=StatevectorSampler()), feature_map=fm)
            qsvc = QSVC(quantum_kernel=q_ker, C=args.qsvm_C)
            qsvc.fit(X_tr_q, y_train)
            y_score = qsvc.decision_function(X_te_q)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)
            y_pred = qsvc.predict(X_te_q)
            m = eval_metrics(y_test, y_score, y_pred)
            for k in m: stats["QSVM"][k].append(m[k])

        # --- SVM ---
        svc = SVC(kernel="rbf", probability=True, C=args.svm_C, random_state=42+run)
        svc.fit(X_train, y_train)
        y_score = svc.predict_proba(X_test)[:, 1]
        y_pred = svc.predict(X_test)
        m = eval_metrics(y_test, y_score, y_pred)
        for k in m: stats["SVM"][k].append(m[k])

        # --- MLP ---
        mlp = MLPClassifier(hidden_layer_sizes=args.mlp_layers, max_iter=args.mlp_iter, random_state=42+run)
        mlp.fit(X_train, y_train)
        y_score = mlp.predict_proba(X_test)[:, 1]
        y_pred = mlp.predict(X_test)
        m = eval_metrics(y_test, y_score, y_pred)
        for k in m: stats["MLP"][k].append(m[k])

    # Log to file and print
    with open(log_file, "a") as f:
        header = f"\nTest Corpus: {test_name} (Train: {os.path.basename(train_path)})\n" + "="*50 + "\n"
        print(header)
        f.write(header)
        
        for model in model_list:
            res_str = f"Model: {model}\n"
            for met in ["acc", "f1", "auc", "eer", "tpr@5"]:
                mean = np.mean(stats[model][met])
                std = np.std(stats[model][met])
                res_str += f"  {met.upper():8}: {mean:.4f} ± {std:.4f}\n"
            print(res_str)
            f.write(res_str + "\n")

def main(args):
    # List of test datasets for cross-corpus evaluation
    # Edit these names to match your actual files (a.npz, b.npz, c.npz, etc.)
    test_datasets = [
        args.test_npz, # The primary one from argument
        "a.npz", 
        "b.npz", 
        "c.npz"
    ]
    
    log_file = "cross_corpus_results.txt"
    # Clear log file at start
    with open(log_file, "w") as f:
        f.write("CROSS CORPUS EVALUATION RESULTS\n" + "#"*40 + "\n")

    for test_path in test_datasets:
        if os.path.exists(test_path):
            run_evaluation(args.train_npz, test_path, args, log_file)
        else:
            print(f"Skipping {test_path}: File not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, default="add_embeddings.npz")
    parser.add_argument("--test_npz", type=str, default="asv_embeddings.npz")
    parser.add_argument("--shared_dim", type=int, default=12)
    parser.add_argument("--use_qsvm", action="store_true")
    parser.add_argument("--qsvm_reps", type=int, default=2)
    parser.add_argument("--qsvm_C", type=float, default=10.0)
    parser.add_argument("--svm_C", type=float, default=50.0)
    parser.add_argument("--mlp_layers", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--mlp_iter", type=int, default=500)
    main(parser.parse_args())