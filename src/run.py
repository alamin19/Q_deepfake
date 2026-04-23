import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, det_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from scipy.special import softmax

# Qiskit imports for QSVM
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.primitives import  StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


def compute_eer(y_true, y_score):
    """Compute Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    # Find the threshold where FPR = FNR
    abs_diff = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diff)
    
    eer = (fpr[min_index] + fnr[min_index]) / 2
    eer_threshold = thresholds[min_index]
    
    return eer, eer_threshold


def compute_ece(y_true, y_score, n_bins=10):
    """Compute Expected Calibration Error (ECE)"""
    # Bin predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_score > bin_lower) & (y_score <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_score[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def eval_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0
    
    # Compute EER and ECE
    try:
        eer, eer_threshold = compute_eer(y_true, y_score)
    except:
        eer = 1.0
        eer_threshold = 0.5
    
    try:
        ece = compute_ece(y_true, y_score)
    except:
        ece = 1.0
    
    cm = confusion_matrix(y_true, y_pred)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "eer": eer, "eer_threshold": eer_threshold, "ece": ece, "cm": cm
    }


def diagnose_kernel(kernel_matrix, y_train, fold_idx):
    """Diagnose potential issues with quantum kernel"""
    print(f"\n[FOLD {fold_idx}] Kernel Diagnostics:")
    print(f"  Kernel shape: {kernel_matrix.shape}")
    print(f"  Kernel mean: {np.mean(kernel_matrix):.4f}")
    print(f"  Kernel std: {np.std(kernel_matrix):.4f}")
    print(f"  Kernel min: {np.min(kernel_matrix):.4f}")
    print(f"  Kernel max: {np.max(kernel_matrix):.4f}")
    
    # Check for kernel collapse (all values too similar)
    if np.std(kernel_matrix) < 0.01:
        print("  ⚠️  WARNING: Kernel collapse detected! All values too similar.")
        print("     → Try: reducing dimensions, changing feature map, or adjusting reps")
    
    # Check diagonal (should be close to 1 for normalized kernels)
    diag_mean = np.mean(np.diag(kernel_matrix))
    print(f"  Diagonal mean: {diag_mean:.4f}")
    if abs(diag_mean - 1.0) > 0.1:
        print(f"  ⚠️  WARNING: Diagonal not near 1.0")
        print("     → Kernel may not be properly normalized")
    
    # Check class separability in kernel space
    kernel_same_class = []
    kernel_diff_class = []
    for i in range(len(y_train)):
        for j in range(i+1, len(y_train)):
            if y_train[i] == y_train[j]:
                kernel_same_class.append(kernel_matrix[i, j])
            else:
                kernel_diff_class.append(kernel_matrix[i, j])
    
    if kernel_same_class and kernel_diff_class:
        same_mean = np.mean(kernel_same_class)
        diff_mean = np.mean(kernel_diff_class)
        print(f"  Same class kernel avg: {same_mean:.4f}")
        print(f"  Diff class kernel avg: {diff_mean:.4f}")
        print(f"  Separability gap: {same_mean - diff_mean:.4f}")
        
        if abs(same_mean - diff_mean) < 0.05:
            print("  ⚠️  WARNING: Poor class separability in kernel space!")
            print("     → Classes are not distinguishable in quantum feature space")
            print("     → Try: different feature map, more reps, or different scaling")


def plot_roc_curves(all_y_true, all_y_scores, model_names, save_path=None):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (y_true, y_scores, name) in enumerate(zip(all_y_true, all_y_scores, model_names)):
        # Concatenate all folds
        y_true_all = np.concatenate(y_true)
        y_scores_all = np.concatenate(y_scores)
        
        fpr, tpr, _ = roc_curve(y_true_all, y_scores_all)
        auc = roc_auc_score(y_true_all, y_scores_all)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                linewidth=2, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_roc.png'), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_det_curves(all_y_true, all_y_scores, model_names, save_path=None):
    """Plot DET curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (y_true, y_scores, name) in enumerate(zip(all_y_true, all_y_scores, model_names)):
        # Concatenate all folds
        y_true_all = np.concatenate(y_true)
        y_scores_all = np.concatenate(y_scores)
        
        fpr, fnr, _ = det_curve(y_true_all, y_scores_all, pos_label=1)
        
        # Convert to percentages
        fpr_percent = fpr * 100
        fnr_percent = fnr * 100
        
        plt.plot(fpr_percent, fnr_percent, label=name, 
                linewidth=2, color=colors[idx])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate (%)', fontsize=12)
    plt.ylabel('False Negative Rate (%)', fontsize=12)
    plt.title('DET Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    
    # Add diagonal line
    min_val = min(plt.xlim()[0], plt.ylim()[0])
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.5)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_det.png'), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_calibration_curve(all_y_true, all_y_scores, model_names, n_bins=10, save_path=None):
    """Plot calibration curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (y_true, y_scores, name) in enumerate(zip(all_y_true, all_y_scores, model_names)):
        # Concatenate all folds
        y_true_all = np.concatenate(y_true)
        y_scores_all = np.concatenate(y_scores)
        
        # Compute calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores_all > bin_lower) & (y_scores_all <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                accuracies.append(np.mean(y_true_all[in_bin]))
        
        plt.plot(bin_centers, accuracies, 'o-', label=name, 
                linewidth=2, markersize=8, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_calibration.png'), dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_score_distributions(all_y_true, all_y_scores, model_names, save_path=None):
    """Plot score distributions for bonafide vs spoof"""
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (y_true, y_scores, name) in enumerate(zip(all_y_true, all_y_scores, model_names)):
        # Concatenate all folds
        y_true_all = np.concatenate(y_true)
        y_scores_all = np.concatenate(y_scores)
        
        # Separate scores by class
        bonafide_scores = y_scores_all[y_true_all == 0]
        spoof_scores = y_scores_all[y_true_all == 1]
        
        # Plot distributions
        axes[idx].hist(bonafide_scores, bins=30, alpha=0.6, label='Bonafide', color='green', density=True)
        axes[idx].hist(spoof_scores, bins=30, alpha=0.6, label='Spoof', color='red', density=True)
        
        axes[idx].set_xlabel('Score', fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'{name}\nScore Distribution', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_score_dist.png'), dpi=300, bbox_inches='tight')
    
    return fig


def main(args):
    data = np.load(args.in_npz)
    X = data["X"]
    y = data["y"]

    print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")

    # Subsample to equal size for all models
    if args.sample_size > 0 and args.sample_size < len(X):
        # Stratified sampling
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y, train_size=args.sample_size, stratify=y, random_state=42
        )
        print(f"Subsampled to {args.sample_size} samples")

    print(f"Using {X.shape[0]} samples for all models")
    print(f"Class distribution: Bonafide={np.sum(y==0)}, Spoof={np.sum(y==1)}")

    # Preprocessing - standardize first
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA reduction - ALL MODELS USE SAME DIMENSIONS for fairness
    pca = PCA(args.shared_dim)
    Xr = pca.fit_transform(Xs)
    print(f"After PCA: {Xr.shape} - ALL models will use {args.shared_dim} features")
    print(f"PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Storage for results
    qsvm_results = []
    svm_results = []
    mlp_results = []

    qsvm_cms = []
    svm_cms = []
    mlp_cms = []

    # Storage for ROC/DET curves
    qsvm_y_true = []
    qsvm_y_scores = []
    svm_y_true = []
    svm_y_scores = []
    mlp_y_true = []
    mlp_y_scores = []

    print("\n" + "="*60)
    print("STARTING 5-FOLD CROSS-VALIDATION")
    print(f"All models use {args.shared_dim} features for fair comparison")
    print("="*60)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xr, y), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/5")
        print(f"{'='*60}")

        # Split data - SAME DATA FOR ALL MODELS
        X_train, X_test = Xr[train_idx], Xr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Train class dist: Bonafide={np.sum(y_train==0)}, Spoof={np.sum(y_train==1)}")

        ########################
        # QSVM
        ########################
        if args.use_qsvm:
            print("\n--- Training QSVM ---")
            try:
                # Scale for quantum circuit [0, π] or [-π, π]
                if args.qsvm_scaler == "minmax":
                    q_scaler = MinMaxScaler(feature_range=(0, np.pi))
                elif args.qsvm_scaler == "minmax_sym":
                    q_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
                elif args.qsvm_scaler == "robust":
                    q_scaler = RobustScaler()
                    X_train_q_tmp = q_scaler.fit_transform(X_train)
                    X_test_q_tmp = q_scaler.transform(X_test)
                    # Map to [0, π]
                    X_train_q = (X_train_q_tmp - X_train_q_tmp.min()) / \
                                (X_train_q_tmp.max() - X_train_q_tmp.min() + 1e-10) * np.pi
                    X_test_q = (X_test_q_tmp - X_test_q_tmp.min()) / \
                               (X_test_q_tmp.max() - X_test_q_tmp.min() + 1e-10) * np.pi
                
                if args.qsvm_scaler in ["minmax", "minmax_sym"]:
                    X_train_q = q_scaler.fit_transform(X_train)
                    X_test_q = q_scaler.transform(X_test)
                
                print(f"Quantum features scaled to range: [{np.min(X_train_q):.3f}, {np.max(X_train_q):.3f}]")
                
                # Create quantum feature map based on user choice
                if args.qsvm_feature_map == "ZZ":
                    feature_map = ZZFeatureMap(
                        feature_dimension=args.shared_dim,
                        reps=args.qsvm_reps,
                        entanglement=args.qsvm_entanglement
                    )
                elif args.qsvm_feature_map == "Z":
                    feature_map = ZFeatureMap(
                        feature_dimension=args.shared_dim,
                        reps=args.qsvm_reps
                    )
                elif args.qsvm_feature_map == "Pauli":
                    feature_map = PauliFeatureMap(
                        feature_dimension=args.shared_dim,
                        reps=args.qsvm_reps,
                        paulis=['Z', 'ZZ'],
                        entanglement=args.qsvm_entanglement
                    )
                elif args.qsvm_feature_map == "PauliExpanded":
                    feature_map = PauliFeatureMap(
                        feature_dimension=args.shared_dim,
                        reps=args.qsvm_reps,
                        paulis=['Z', 'ZZ', 'ZZZ'],
                        entanglement=args.qsvm_entanglement
                    )
                
                print(f"Feature Map: {args.qsvm_feature_map}, Reps: {args.qsvm_reps}, "
                      f"Entanglement: {args.qsvm_entanglement}")
                print(f"Quantum circuit depth: ~{feature_map.num_parameters} parameters")
                
                # Create quantum kernel
                if args.qsvm_sampler == "statevector":
                    sampler = StatevectorSampler()
                else:  # default Sampler
                    sampler = StatevectorSampler()
                
                fidelity = ComputeUncompute(sampler=sampler)
                qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

                # Compute kernel matrix for diagnostics
                if args.diagnose:
                    print("\nComputing kernel matrix for diagnostics...")
                    train_kernel = qkernel.evaluate(x_vec=X_train_q)
                    diagnose_kernel(train_kernel, y_train, fold_idx)

                # Create and train QSVC
                qsvc = QSVC(
                    quantum_kernel=qkernel,
                    C=args.qsvm_C,
                    class_weight='balanced' if args.qsvm_balanced else None
                )
                
                print(f"Training QSVC with C={args.qsvm_C}, balanced={args.qsvm_balanced}...")
                qsvc.fit(X_train_q, y_train)
                
                # Predict
                y_pred_q = qsvc.predict(X_test_q)
                y_score_q = qsvc.decision_function(X_test_q)
                
                # Normalize decision function
                if np.max(y_score_q) != np.min(y_score_q):
                    y_score_q = (y_score_q - y_score_q.min()) / (y_score_q.max() - y_score_q.min() + 1e-10)
                else:
                    print("  ⚠️  WARNING: Decision function has zero variance!")
                    y_score_q = np.ones_like(y_score_q) * 0.5
                
                qsvm_results.append(eval_metrics(y_test, y_pred_q, y_score_q))
                qsvm_cms.append(qsvm_results[-1]['cm'])
                qsvm_y_true.append(y_test)
                qsvm_y_scores.append(y_score_q)
                
                print(f"QSVM - Acc: {qsvm_results[-1]['acc']:.4f}, F1: {qsvm_results[-1]['f1']:.4f}, "
                      f"AUC: {qsvm_results[-1]['auc']:.4f}, EER: {qsvm_results[-1]['eer']:.4f}, "
                      f"ECE: {qsvm_results[-1]['ece']:.4f}")
                
                # Check for degenerate predictions
                if len(np.unique(y_pred_q)) == 1:
                    print(f"  ⚠️  CRITICAL: QSVM predicting only class {y_pred_q[0]}!")
                    print("     → Model is degenerate. Try different hyperparameters.")
                
            except Exception as e:
                print(f"QSVM Error: {e}")
                import traceback
                traceback.print_exc()
                # Add dummy result
                qsvm_results.append({
                    'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0,
                    'eer': 1.0, 'eer_threshold': 0.5, 'ece': 1.0,
                    'cm': np.zeros((2, 2))
                })
                qsvm_cms.append(np.zeros((2, 2)))
                qsvm_y_true.append(y_test)
                qsvm_y_scores.append(np.ones_like(y_test) * 0.5)

        ########################
        # SVM (uses same features as QSVM)
        ########################
        print("\n--- Training SVM ---")
        svc = SVC(
            kernel="rbf", 
            probability=True, 
            random_state=42,
            C=args.svm_C,
            class_weight='balanced' if args.svm_balanced else None
        )
        svc.fit(X_train, y_train)

        y_pred_s = svc.predict(X_test)
        y_score_s = svc.predict_proba(X_test)[:, 1]
        svm_results.append(eval_metrics(y_test, y_pred_s, y_score_s))
        svm_cms.append(svm_results[-1]['cm'])
        svm_y_true.append(y_test)
        svm_y_scores.append(y_score_s)

        print(f"SVM  - Acc: {svm_results[-1]['acc']:.4f}, F1: {svm_results[-1]['f1']:.4f}, "
              f"AUC: {svm_results[-1]['auc']:.4f}, EER: {svm_results[-1]['eer']:.4f}, "
              f"ECE: {svm_results[-1]['ece']:.4f}")

        ########################
        # MLP (uses same features as QSVM)
        ########################
        print("\n--- Training MLP ---")
        mlp = MLPClassifier(
            hidden_layer_sizes=args.mlp_layers, 
            max_iter=args.mlp_iter, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp.fit(X_train, y_train)

        y_pred_m = mlp.predict(X_test)
        y_score_m = mlp.predict_proba(X_test)[:, 1]
        mlp_results.append(eval_metrics(y_test, y_pred_m, y_score_m))
        mlp_cms.append(mlp_results[-1]['cm'])
        mlp_y_true.append(y_test)
        mlp_y_scores.append(y_score_m)

        print(f"MLP  - Acc: {mlp_results[-1]['acc']:.4f}, F1: {mlp_results[-1]['f1']:.4f}, "
              f"AUC: {mlp_results[-1]['auc']:.4f}, EER: {mlp_results[-1]['eer']:.4f}, "
              f"ECE: {mlp_results[-1]['ece']:.4f}")

    ########################
    # Aggregate Results
    ########################
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)

    def print_cv_results(results, model_name):
        metrics = ['acc', 'prec', 'rec', 'f1', 'auc', 'eer', 'ece']
        print(f"\n{model_name} Results:")
        print("-" * 50)
        
        for metric in metrics:
            values = [r[metric] for r in results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")
        
        return {m: (np.mean([r[m] for r in results]), np.std([r[m] for r in results])) 
                for m in metrics}

    if args.use_qsvm:
        qsvm_stats = print_cv_results(qsvm_results, "QSVM")
    svm_stats = print_cv_results(svm_results, "SVM")
    mlp_stats = print_cv_results(mlp_results, "MLP")

    ########################
    # Recommendations
    ########################
    if args.use_qsvm and qsvm_stats['acc'][0] < 0.6:
        print("\n" + "="*60)
        print("⚠️  QSVM PERFORMANCE IS POOR - RECOMMENDATIONS:")
        print("="*60)
        print("1. REDUCE DIMENSIONS: Try --shared_dim 4 or --shared_dim 6")
        print("2. SIMPLIFY CIRCUIT: Try --qsvm_reps 1 or --qsvm_reps 2")
        print("3. CHANGE FEATURE MAP: Try --qsvm_feature_map Z (simplest)")
        print("4. CHANGE ENTANGLEMENT: Try --qsvm_entanglement linear")
        print("5. ADJUST SCALING: Try --qsvm_scaler minmax_sym")
        print("6. INCREASE C: Try --qsvm_C 50 or --qsvm_C 100")
        print("7. USE BALANCE: Add --qsvm_balanced flag")
        print("8. ENABLE DIAGNOSTICS: Add --diagnose to see kernel issues")
        print("\nExample command:")
        print("  python run.py --use_qsvm --shared_dim 6 --qsvm_reps 2 \\")
        print("    --qsvm_feature_map Z --qsvm_C 50 --qsvm_balanced --diagnose")

    ########################
    # Comparison Table
    ########################
    print("\n" + "="*60)
    print("MODEL COMPARISON (Mean ± Std)")
    print("="*60)

    comparison_data = []
    
    if args.use_qsvm:
        comparison_data.append({
            "Model": "QSVM",
            "Accuracy": f"{qsvm_stats['acc'][0]:.4f} ± {qsvm_stats['acc'][1]:.4f}",
            "F1-Score": f"{qsvm_stats['f1'][0]:.4f} ± {qsvm_stats['f1'][1]:.4f}",
            "AUC": f"{qsvm_stats['auc'][0]:.4f} ± {qsvm_stats['auc'][1]:.4f}",
            "EER": f"{qsvm_stats['eer'][0]:.4f} ± {qsvm_stats['eer'][1]:.4f}",
            "ECE": f"{qsvm_stats['ece'][0]:.4f} ± {qsvm_stats['ece'][1]:.4f}"
        })
    
    comparison_data.append({
        "Model": "SVM",
        "Accuracy": f"{svm_stats['acc'][0]:.4f} ± {svm_stats['acc'][1]:.4f}",
        "F1-Score": f"{svm_stats['f1'][0]:.4f} ± {svm_stats['f1'][1]:.4f}",
        "AUC": f"{svm_stats['auc'][0]:.4f} ± {svm_stats['auc'][1]:.4f}",
        "EER": f"{svm_stats['eer'][0]:.4f} ± {svm_stats['eer'][1]:.4f}",
        "ECE": f"{svm_stats['ece'][0]:.4f} ± {svm_stats['ece'][1]:.4f}"
    })
    
    comparison_data.append({
        "Model": "MLP",
        "Accuracy": f"{mlp_stats['acc'][0]:.4f} ± {mlp_stats['acc'][1]:.4f}",
        "F1-Score": f"{mlp_stats['f1'][0]:.4f} ± {mlp_stats['f1'][1]:.4f}",
        "AUC": f"{mlp_stats['auc'][0]:.4f} ± {mlp_stats['auc'][1]:.4f}",
        "EER": f"{mlp_stats['eer'][0]:.4f} ± {mlp_stats['eer'][1]:.4f}",
        "ECE": f"{mlp_stats['ece'][0]:.4f} ± {mlp_stats['ece'][1]:.4f}"
    })
    
    comp_df = pd.DataFrame(comparison_data)
    print("\n" + comp_df.to_string(index=False))

    ########################
    # Plot All Visualizations
    ########################
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Prepare data for plotting
    all_y_true = []
    all_y_scores = []
    model_names = []
    
    if args.use_qsvm:
        all_y_true.append(qsvm_y_true)
        all_y_scores.append(qsvm_y_scores)
        model_names.append("QSVM")
    
    all_y_true.append(svm_y_true)
    all_y_scores.append(svm_y_scores)
    model_names.append("SVM")
    
    all_y_true.append(mlp_y_true)
    all_y_scores.append(mlp_y_scores)
    model_names.append("MLP")
    
    # Plot ROC Curves
    print("Plotting ROC curves...")
    plot_roc_curves(all_y_true, all_y_scores, model_names, args.save_plot)
    
    # Plot DET Curves
    print("Plotting DET curves...")
    plot_det_curves(all_y_true, all_y_scores, model_names, args.save_plot)
    
    # Plot Calibration Curves
    print("Plotting calibration curves...")
    plot_calibration_curve(all_y_true, all_y_scores, model_names, save_path=args.save_plot)
    
    # Plot Score Distributions
    print("Plotting score distributions...")
    plot_score_distributions(all_y_true, all_y_scores, model_names, save_path=args.save_plot)

    ########################
    # Confusion Matrices
    ########################
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    if num_models == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if args.use_qsvm:
        avg_cm_q = np.mean(qsvm_cms, axis=0).astype(int)
        df_q = pd.DataFrame(avg_cm_q, 
                           index=["bonafide", "spoof"], 
                           columns=["pred_bonafide", "pred_spoof"])
        sns.heatmap(df_q, annot=True, fmt="d", ax=axes[plot_idx], cmap="Blues")
        axes[plot_idx].set_title(
            f"QSVM Confusion Matrix\n"
            f"Acc: {qsvm_stats['acc'][0]:.3f}, EER: {qsvm_stats['eer'][0]:.3f}"
        )
        plot_idx += 1
    
    avg_cm_svm = np.mean(svm_cms, axis=0).astype(int)
    df_svm = pd.DataFrame(avg_cm_svm, 
                         index=["bonafide", "spoof"], 
                         columns=["pred_bonafide", "pred_spoof"])
    sns.heatmap(df_svm, annot=True, fmt="d", ax=axes[plot_idx], cmap="Blues")
    axes[plot_idx].set_title(
        f"SVM Confusion Matrix\n"
        f"Acc: {svm_stats['acc'][0]:.3f}, EER: {svm_stats['eer'][0]:.3f}"
    )
    plot_idx += 1
    
    avg_cm_mlp = np.mean(mlp_cms, axis=0).astype(int)
    df_mlp = pd.DataFrame(avg_cm_mlp, 
                         index=["bonafide", "spoof"], 
                         columns=["pred_bonafide", "pred_spoof"])
    sns.heatmap(df_mlp, annot=True, fmt="d", ax=axes[plot_idx], cmap="Blues")
    axes[plot_idx].set_title(
        f"MLP Confusion Matrix\n"
        f"Acc: {mlp_stats['acc'][0]:.3f}, EER: {mlp_stats['eer'][0]:.3f}"
    )
    
    plt.tight_layout()
    
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to {args.save_plot}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair comparison of QSVM vs Classical ML")
    
    # Data arguments
    parser.add_argument("--in_npz", type=str, default="asvspoof5_flac_embeddings.npz")
    parser.add_argument("--sample_size", type=int, default=500,
                       help="Number of samples to use for all models (0 = use all)")
    
    # FAIR COMPARISON: All models use same dimensions
    parser.add_argument("--shared_dim", type=int, default=12,
                       help="Feature dimensions for ALL models (fair comparison)")
    
    # QSVM arguments
    parser.add_argument("--use_qsvm", action="store_true", help="Enable QSVM classifier")
    parser.add_argument("--qsvm_feature_map", type=str, default="ZZ",
                       choices=["ZZ", "Z", "Pauli", "PauliExpanded"],
                       help="Quantum feature map type")
    parser.add_argument("--qsvm_reps", type=int, default=3,
                       help="Number of repetitions in quantum feature map")
    parser.add_argument("--qsvm_entanglement", type=str, default="full",
                       choices=["full", "linear", "circular", "sca"],
                       help="Entanglement pattern for quantum circuit")
    parser.add_argument("--qsvm_C", type=float, default=10.0,
                       help="Regularization parameter for QSVM")
    parser.add_argument("--qsvm_balanced", action="store_true",
                       help="Use balanced class weights for QSVM")
    parser.add_argument("--qsvm_scaler", type=str, default="minmax",
                       choices=["minmax", "minmax_sym", "robust"],
                       help="Scaling method for quantum features")
    parser.add_argument("--qsvm_sampler", type=str, default="sampler",
                       choices=["sampler", "statevector"],
                       help="Qiskit sampler type")
    
    # Diagnostics
    parser.add_argument("--diagnose", action="store_true",
                       help="Enable detailed kernel diagnostics (slower)")
    
    # SVM arguments
    parser.add_argument("--svm_C", type=float, default=50.0,
                       help="Regularization parameter for SVM")
    parser.add_argument("--svm_balanced", action="store_true",
                       help="Use balanced class weights for SVM")
    
    # MLP arguments
    parser.add_argument("--mlp_layers", type=int, nargs="+", default=[128, 64],
                       help="Hidden layer sizes for MLP")
    parser.add_argument("--mlp_iter", type=int, default=500,
                       help="Max iterations for MLP")
    
    # Output
    parser.add_argument("--save_plot", type=str, default="", 
                       help="Path to save plots (will create multiple files)")
    
    args = parser.parse_args()
    main(args)