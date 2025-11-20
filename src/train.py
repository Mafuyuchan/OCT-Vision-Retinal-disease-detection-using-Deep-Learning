import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data import OCTDataset
from models.vgg16_transfer import VGG16Transfer
from models.resnet50_transfer import ResNet50Transfer
from utils import save_checkpoint
from train_utils import train_one_epoch  # assuming you have this


# ---------------------------------------
# Validation / Evaluation Loop
# ---------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return running_loss / total, correct / total


# ---------------------------------------
# Main Function
# ---------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='dataset root (with train/val folders)')
    parser.add_argument('--model', choices=['vgg16', 'resnet50'], default='resnet50')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    device = torch.device(args.device)

    # ---------------------------------------
    # LOAD DATA
    # ---------------------------------------
    dataset = OCTDataset(args.data, img_size=args.img_size, batch_size=args.batch_size)
    train_loader, val_loader, class_names = dataset.make_loaders()

    # ---------------------------------------
    # MODEL
    # ---------------------------------------
    if args.model == 'vgg16':
        model = VGG16Transfer(num_classes=args.num_classes)
    else:
        model = ResNet50Transfer(num_classes=args.num_classes)

    model = model.to(device)

    # ---------------------------------------
    # LOSS, OPTIMIZER, LR SCHEDULER
    # ---------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_acc = 0.0

    # ---------------------------------------
    # TRAINING LOOP
    # ---------------------------------------
    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{args.epochs} "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint)
            print(f">> Saved BEST checkpoint to {args.checkpoint}")

    print("\nTraining finished. Best val accuracy:", best_acc)


# ---------------------------------------
if __name__ == "__main__":
    main()
