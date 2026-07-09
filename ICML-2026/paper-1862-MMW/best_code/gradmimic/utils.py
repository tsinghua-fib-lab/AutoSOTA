import torch
from tqdm.autonotebook import tqdm


def evaluate(model, test_loader, device, use_noisy_labels=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, inputs, true_labels, noisy_labels in tqdm(test_loader, leave=False, desc="Eval"):
            labels = noisy_labels if use_noisy_labels else true_labels
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total
