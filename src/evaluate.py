# src/evaluate.py

import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.utils import save_image

from gradcam import GradCAM  # your gradcam class


def evaluate_model(model, dataloader, device):
    """
    Returns:
        accuracy (float), labels (list[int]), preds (list[int]), probs (list[list[float]])
    """
    model.eval()
    total, correct = 0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)

            preds = probabilities.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probabilities.cpu().tolist())

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels(class_names,
                    rotation=45,
                    ha="right"),
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Print numbers on heatmap
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_gradcam(model, dataloader, class_names, device, save_dir, num_images=3):
    """
    Saves GradCAM heatmaps for first few images in dataloader.
    """
    model.eval()

    # model must have model.target_layer
    if not hasattr(model, "target_layer"):
        print("[WARN] Model has no target_layer attribute. Skipping Grad-CAM.")
        return

    cam = GradCAM(model, model.target_layer)

    saved = 0
    for images, labels in dataloader:
        images = images.to(device)

        # Only process one image at a time for Grad-CAM
        for i in range(images.size(0)):
            img = images[i:i+1]
            label = labels[i].item()

            heatmap = cam(img)   # (H, W) 0..1

            heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            original = img[0].permute(1, 2, 0).cpu().numpy()
            original = (original - original.min()) / (original.max() - original.min() + 1e-6)

            overlay = (0.5 * original + 0.5 * (heatmap / 255.0))

            out_path = os.path.join(save_dir, f"gradcam_{saved}.png")
            plt.imsave(out_path, overlay)

            print(f"Saved Grad-CAM → {out_path}")

            saved += 1
            if saved >= num_images:
                return


def evaluate(
    model,
    dataloader,
    class_names,
    device="cuda",
    save_dir="evaluation_results"
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- MAIN EVALUATION ----
    accuracy, labels, preds, probs = evaluate_model(model, dataloader, device)

    # ---- CLASSIFICATION REPORT ----
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True
    )

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "classification_report": report
        }, f, indent=4)

    print("\n✔ Accuracy:", accuracy)
    print("\n✔ Classification Report saved → metrics.json")

    # ---- CONFUSION MATRIX ----
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path)
    print("✔ Confusion matrix saved →", cm_path)

    # ---- GRAD-CAM ----
    run_gradcam(model, dataloader, class_names, device, save_dir)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    print("Run this script from train/eval pipeline.")

