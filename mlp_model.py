"""
Multi-Layer Perceptron (MLP) Models for Tabular Data Classification and Regression
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MLPTrainer(nn.Module):
    def __init__(self, model_type, hidden_dims, dropout_rate, learning_rate, batch_size, random_state):
        super().__init__()
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Output dimension
        output_dim = 1 if model_type == 'regressor' else 1

        # Build model
        layers = []
        dims = [None] + hidden_dims + [output_dim]  # input_dim unknown yet
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.model = None  # build later with input dim known

    def _build_model(self, input_dim):
        dims = [input_dim] + self.hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ]
        layers.append(nn.Linear(dims[-1], 1))
        if self.model_type == 'classifier':
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def train(self, X, y, epochs=100, validation_split=0.2, early_stopping_patience=10, verbose=False):
        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        self._build_model(X.shape[1])
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss() if self.model_type == 'classifier' else nn.MSELoss()

        best_val_loss = float('inf')
        patience = 0
        self.history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            permutation = torch.randperm(X_train.size(0))
            epoch_loss = 0
            for i in range(0, X_train.size(0), self.batch_size):
                indices = permutation[i:i+self.batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val)
                val_loss = criterion(val_preds, y_val).item()

            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            preds = self.model(X_test)
            if self.model_type == 'classifier':
                y_pred = (preds > 0.5).float()
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                return {'accuracy': acc, 'f1': f1}
            else:
                mse = nn.MSELoss()(preds, y_test).item()
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, preds.numpy())
                return {'rmse': rmse, 'r2': r2}

    def plot_training_history(self):
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True)
        plt.show()
