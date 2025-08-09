"""
Multi-Layer Perceptron (MLP) Models for Tabular Data Classification and Regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for Binary Classification
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPRegressor(nn.Module):
    """
    Multi-Layer Perceptron for Regression
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPTrainer:
    """
    Trainer class for MLP models
    """
    
    def __init__(self, model_type: str = 'classifier', hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3, learning_rate: float = 0.001, 
                 batch_size: int = 64, random_state: int = 42):
        
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'loss': [], 'val_loss': []}
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training and validation
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=self.random_state
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100,
             validation_split: float = 0.2, early_stopping_patience: int = 10,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the MLP model
        """
        input_dim = X_train.shape[1]
        
        # Initialize model
        if self.model_type == 'classifier':
            self.model = MLPClassifier(input_dim, self.hidden_dims, self.dropout_rate)
            self.criterion = nn.BCELoss()
        else:  # regressor
            self.model = MLPRegressor(input_dim, self.hidden_dims, self.dropout_rate)
            self.criterion = nn.MSELoss()
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(X_train, y_train, validation_split)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                self.model.load_state_dict(best_model_state)
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        return self.training_history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy().flatten()
        
        if self.model_type == 'classifier':
            # Binary classification metrics
            y_pred = (predictions > 0.5).astype(int)
            y_pred_proba = predictions
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
        else:
            # Regression metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions),
                'mse': mean_squared_error(y_test, predictions)
            }
        
        return metrics
    
    def plot_training_history(self) -> None:
        """
        Plot training and validation loss
        """
        if not self.training_history['loss']:
            raise ValueError("No training history available")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MLP Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot predictions vs actual values (for regression)
        """
        if self.model_type != 'regressor':
            print("Prediction plots are only available for regression models")
            return
        
        if self.model is None:
            raise ValueError("Model must be trained before plotting predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = self.model(X_test_tensor).cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('MLP: Predictions vs Actual Values')
        plt.tight_layout()
        plt.show()
        
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('MLP: Residual Plot')
        plt.tight_layout()
        plt.show()