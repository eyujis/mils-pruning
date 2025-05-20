import torch

def test(model, test_loader, device):
    """
    Evaluates the model on the test set and prints accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    test_loader : DataLoader
        DataLoader for the test set.
    device : torch.device
        The device to run evaluation on.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100 * correct / len(test_loader.dataset)
    
    return acc
