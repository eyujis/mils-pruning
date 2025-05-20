import os
import torch

def train(model, train_loader, val_loader, optimizer, criterion, results, early_stopping, epochs=10, device=None, experiment_id="000"):
    """
    Trains a model with early stopping and saves the best model based on validation loss.

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
        Loss function.
    results : object or None
        Tracker for logging results (must implement .update()).
    early_stopping : EarlyStopping
        Early stopping callback.
    epochs : int
        Maximum number of training epochs.
    device : torch.device
        Device to run training on.
    experiment_id : str
        Identifier for saving model weights.
    """
    # Create folder to save weights
    experiment_dir = f"saved_weights/experiment_{experiment_id}"
    os.makedirs(experiment_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss, correct = 0, 0

        # ------------------- Training phase -------------------
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = 100 * correct / len(train_loader.dataset)

        # ------------------- Validation phase -------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / len(val_loader.dataset)

        # ------------------- Logging -------------------
        if results is not None:
            results.update(avg_train_loss, avg_val_loss, train_acc, val_acc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # ------------------- Early Stopping -------------------
        if early_stopping(avg_val_loss, model, epoch):
            print(f"Early stopping triggered at epoch {epoch+1}!")
            print(f"Best model was from epoch {early_stopping.best_epoch+1}.")
            break

    # ------------------- Save Best Model -------------------
    best_model_path = os.path.join(experiment_dir, "best_model.pt")
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
