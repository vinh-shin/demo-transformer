import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(labels, num_classes: int = 3, device=None):
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    weights = torch.tensor(class_weights_np, dtype=torch.float)
    if device is not None:
        weights = weights.to(device)
    return weights


def _target_names(label_names):
    return [label_names[i] for i in sorted(label_names)]


def print_classification_report(y_true, y_pred, label_names, digits: int = 4):
    print(classification_report(
        y_true, y_pred, target_names=_target_names(label_names), digits=digits
    ))


def plot_confusion_matrix(y_true, y_pred, label_names):
    target_names = _target_names(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_names))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=axes[0])
    axes[0].set_title("Confusion matrix (số lượng)")
    axes[0].set_xlabel("Dự đoán")
    axes[0].set_ylabel("Thực tế")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=axes[1])
    axes[1].set_title("Confusion matrix (chuẩn hoá theo hàng)")
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Thực tế")

    plt.tight_layout()
    plt.show()


def build_misclassified_report(source_df, y_true, y_pred, label_names, text_col="free_text", n_samples=10, seed=42):
    df = source_df.reset_index(drop=True).copy()
    df["true_label"] = [label_names[i] for i in y_true]
    df["pred_label"] = [label_names[i] for i in y_pred]

    misclassified = df[df["true_label"] != df["pred_label"]]
    print(f"Tổng số mẫu sai: {len(misclassified)} / {len(df)} ({len(misclassified)/len(df):.1%})")

    return misclassified[[text_col, "true_label", "pred_label"]].sample(
        min(n_samples, len(misclassified)), random_state=seed
    )
