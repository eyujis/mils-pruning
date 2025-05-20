import os
import torch
from mils_pruning.config import WEIGHTS_DIR


def train(model, train_loader, val_loader, optimizer, criterion, early_stopping, epochs=10, device=None, experiment_id="000"):
    """
    Trains a model using early stopping and saves only the best model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    criterion : torch.nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    early_stopping : EarlyStopping
        Callback to monitor validation loss and stop training early.
    epochs : int
        Maximum number of training epochs.
    device : torch.device
        Device to train on (e.g., torch.device("cuda")).
    experiment_id : str
        Unique identifier used to name the saved model directory.
    """
    # Create a directory to store the best model weights
    experiment_dir = WEIGHTS_DIR / f"experiment_{experiment_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss, correct = 0, 0

        # ------------------- Training phase -------------------
        model.train()  # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate total loss and count correct predictions
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        # Compute average training loss and accuracy
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = 100 * correct / len(train_loader.dataset)

        # ------------------- Validation phase -------------------
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_correct = 0
        with torch.no_grad():  # No gradients needed during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Compute average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / len(val_loader.dataset)

        # Print progress for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # ------------------- Early stopping check -------------------
        if early_stopping(avg_val_loss, model, epoch):
            print(f"Early stopping triggered at epoch {epoch+1}!")
            print(f"Best model was from epoch {early_stopping.best_epoch+1}.")
            break

    # ------------------- Save the best model -------------------
    best_model_path = experiment_dir / "best_model.pt"
    torch.save(early_stopping.best_model_state, best_model_path)
    print(f"Best model saved to: {best_model_path}")

    


class EarlyStopping:
    """
    Stops training if validation loss does not improve after a given number of epochs.
    Saves the best model (with lowest validation loss).
    """

    def __init__(self, patience=5, min_delta=0):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait without improvement before stopping.
        min_delta : float
            Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        """
        Evaluates validation loss and updates best model if improvement is found.

        Returns True to trigger early stopping, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False  # Continue training
