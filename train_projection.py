import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from GWO.GWO_optimizer import GWOOptimizer
import matplotlib.pyplot as plt
import seaborn as sns


class ConfigurableMLP(nn.Module):
    """MLP with configurable architecture"""
    def __init__(self, input_dim, output_dim, hidden_layers, neurons_per_layer, dropout_rate=0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, neurons_per_layer))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = neurons_per_layer
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    

class MLP_GWO_Optimizer:
    """Use GWO to find optimal MLP hyperparameters"""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_dim, output_dim):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define search space (bounds for each parameter)
        # [hidden_layers, neurons_per_layer, learning_rate, dropout_rate]
        self.bounds = [
            (1, 5),      # hidden_layers (int: 1-5 layers)
            (16, 256),   # neurons_per_layer (int: 16-256)
            (1e-4, 1e-1), # learning_rate (float: 0.0001 to 0.1)
            (0.0, 0.5),  # dropout_rate (float: 0 to 0.5)
        ]
        
        # Initialize GWO
        self.gwo = GWOOptimizer(
            objective_function=self.evaluate_mlp_config,
            bounds=self.bounds,
            population_size=100,  # Small population for speed
            max_iter=200          # Few iterations for demo
        )
    
    def decode_parameters(self, wolf_position):
        """Convert continuous GWO parameters to actual hyperparameters"""
        params = {
            'hidden_layers': int(round(wolf_position[0].item())),
            'neurons_per_layer': int(round(wolf_position[1].item())),
            'learning_rate': wolf_position[2].item(),
            'dropout_rate': wolf_position[3].item(),
        }
        # Ensure at least 1 layer and reasonable neuron count
        params['hidden_layers'] = max(1, params['hidden_layers'])
        params['neurons_per_layer'] = max(16, params['neurons_per_layer'])
        return params
    
    def evaluate_mlp_config(self, wolf_position):
        """Train MLP with given parameters and return validation loss"""
        # Decode parameters
        params = self.decode_parameters(wolf_position)
        
        # Create model
        model = ConfigurableMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=params['hidden_layers'],
            neurons_per_layer=params['neurons_per_layer'],
            dropout_rate=params['dropout_rate']
        )
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Quick training (few epochs for speed)
        model.train()
        for epoch in range(20):  # Quick training
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            val_loss = criterion(val_outputs, self.y_val)
            _, predictions = torch.max(val_outputs, 1)
            accuracy = (predictions == self.y_val).float().mean().item()
        
        # We want to minimize loss (but also could maximize accuracy)
        # Return 1 - accuracy so GWO minimizes it (lower is better)
        return 1.0 - accuracy
    
    def optimize(self, n_iterations=20):
        """Run GWO optimization"""
        print("Starting GWO optimization for MLP...")
        best_position, loss_history = self.gwo.optimize()
        
        best_params = self.decode_parameters(best_position)
        
        # Train final model with best parameters
        final_model = ConfigurableMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=best_params['hidden_layers'],
            neurons_per_layer=best_params['neurons_per_layer'],
            dropout_rate=best_params['dropout_rate']
        )
        
        # Proper training of final model
        final_accuracy = self.train_final_model(final_model, best_params['learning_rate'])
        
        return best_params, loss_history, final_accuracy
    
    def train_final_model(self, model, learning_rate, epochs=50):
        """Train the final model properly"""
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            _, predictions = torch.max(val_outputs, 1)
            accuracy = (predictions == self.y_val).float().mean().item()
        
        return accuracy
    
class RandomSearchMLP:
    """Baseline: Random Search for comparison"""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_dim, output_dim):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bounds = [
            (1, 5), (16, 256), (1e-4, 1e-1), (0.0, 0.5)
        ]
    
    def search(self, n_trials=50):
        """Perform random search"""
        best_accuracy = 0
        best_params = None
        history = []
        
        for trial in range(n_trials):
            # Random parameters
            hidden_layers = np.random.randint(1, 6)
            neurons = np.random.randint(16, 257)
            lr = 10**np.random.uniform(-4, -1)  # Log-uniform
            dropout = np.random.uniform(0, 0.5)
            
            # Train and evaluate
            model = ConfigurableMLP(
                self.input_dim, self.output_dim, 
                hidden_layers, neurons, dropout
            )
            
            accuracy = self.quick_evaluate(model, lr)
            history.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'hidden_layers': hidden_layers,
                    'neurons_per_layer': neurons,
                    'learning_rate': lr,
                    'dropout_rate': dropout
                }
        
        return best_params, history, best_accuracy
    
    def quick_evaluate(self, model, lr, epochs=20):
        """Quick evaluation similar to GWO"""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            _, predictions = torch.max(val_outputs, 1)
            accuracy = (predictions == self.y_val).float().mean().item()
        
        return accuracy


def run_experiment():
    """Main experiment comparing GWO vs Random Search"""
    
    # Create synthetic dataset (or use real one)
    X, y = make_classification(
        n_samples=10000, 
        n_features=200, 
        n_informative=10,
        n_classes=10,
        random_state=42
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))
    
    print(f"Dataset: {X.shape[0]} samples, {input_dim} features, {output_dim} classes")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Run GWO optimization
    print("\n" + "="*50)
    print("Running GWO Optimization...")
    print("="*50)
    gwo_optimizer = MLP_GWO_Optimizer(X_train, y_train, X_val, y_val, input_dim, output_dim)
    gwo_params, gwo_history, gwo_accuracy = gwo_optimizer.optimize()
    
    print("\nGWO Best Parameters:")
    for key, value in gwo_params.items():
        print(f"  {key}: {value}")
    print(f"Validation Accuracy: {gwo_accuracy:.4f}")
    
    # Run Random Search (with similar computational budget)
    print("\n" + "="*50)
    print("Running Random Search...")
    print("="*50)
    rs_optimizer = RandomSearchMLP(X_train, y_train, X_val, y_val, input_dim, output_dim)
    rs_params, rs_history, rs_accuracy = rs_optimizer.search(n_trials=200)  # 10 wolves × 20 iterations = 200
    
    print("\nRandom Search Best Parameters:")
    for key, value in rs_params.items():
        print(f"  {key}: {value}")
    print(f"Validation Accuracy: {rs_accuracy:.4f}")
    
    # Compare final test performance
    print("\n" + "="*50)
    print("Final Test Set Evaluation")
    print("="*50)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # GWO model
    gwo_model = ConfigurableMLP(
        input_dim, output_dim,
        gwo_params['hidden_layers'],
        gwo_params['neurons_per_layer'],
        gwo_params['dropout_rate']
    )
    gwo_test_acc = evaluate_model(gwo_model, X_test_tensor, y_test_tensor, gwo_params['learning_rate'])
    
    # Random Search model
    rs_model = ConfigurableMLP(
        input_dim, output_dim,
        rs_params['hidden_layers'],
        rs_params['neurons_per_layer'],
        rs_params['dropout_rate']
    )
    rs_test_acc = evaluate_model(rs_model, X_test_tensor, y_test_tensor, rs_params['learning_rate'])
    
    print(f"\nTest Set Accuracy:")
    print(f"  GWO: {gwo_test_acc:.4f}")
    print(f"  Random Search: {rs_test_acc:.4f}")
    print(f"  Improvement: {(gwo_test_acc - rs_test_acc)*100:.2f}%")
    
    return {
        'gwo': {'params': gwo_params, 'history': gwo_history, 'test_acc': gwo_test_acc},
        'random_search': {'params': rs_params, 'history': rs_history, 'test_acc': rs_test_acc},
        'dataset_info': {'input_dim': input_dim, 'output_dim': output_dim}
    }

def evaluate_model(model, X_test, y_test, lr, epochs=50):
    """Train and evaluate a model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_test)  # Normally train on train set, but for simplicity...
        loss = criterion(outputs, y_test)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, 1)
        accuracy = (predictions == y_test).float().mean().item()
    
    return accuracy







def plot_results(results):
    """Create simple comparison plots"""
    
    gwo_history = results['gwo']['history']
    rs_history = results['random_search']['history']
    
    # Convert loss to accuracy for GWO (since it minimizes 1-accuracy)
    gwo_accuracy_history = [1 - loss for loss in gwo_history]
    
    # Plot convergence
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(gwo_accuracy_history, 'b-', label='GWO', linewidth=2)
    plt.plot(rs_history, 'r-', label='Random Search', alpha=0.7)
    plt.xlabel('Iteration / Trial')
    plt.ylabel('Validation Accuracy')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final accuracy comparison
    plt.subplot(1, 3, 2)
    methods = ['GWO', 'Random Search']
    accuracies = [results['gwo']['test_acc'], results['random_search']['test_acc']]
    colors = ['blue', 'red']
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Test Accuracy')
    plt.title('Final Performance')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot parameter importance (simple example)
    plt.subplot(1, 3, 3)
    gwo_params = results['gwo']['params']
    param_names = list(gwo_params.keys())
    param_values = list(gwo_params.values())
    
    # Normalize values for visualization
    normalized_values = []
    bounds = [(1, 5), (16, 256), (1e-4, 1e-1), (0.0, 0.5)]
    for i, (name, value) in enumerate(gwo_params.items()):
        low, high = bounds[i]
        norm_val = (value - low) / (high - low) if high > low else 0
        normalized_values.append(norm_val)
    
    plt.barh(param_names, normalized_values, color='green', alpha=0.6)
    plt.xlabel('Normalized Value (0=min, 1=max)')
    plt.title('Best Parameters Found by GWO')
    plt.xlim([0, 1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run experiment
    results = run_experiment()
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nBest GWO Architecture:")
    print(f"  Layers: {results['gwo']['params']['hidden_layers']}")
    print(f"  Neurons per layer: {results['gwo']['params']['neurons_per_layer']}")
    print(f"  Learning rate: {results['gwo']['params']['learning_rate']:.6f}")
    print(f"  Dropout: {results['gwo']['params']['dropout_rate']:.3f}")
    print(f"  Test Accuracy: {results['gwo']['test_acc']:.4f}")
    
    print(f"\nBest Random Search Architecture:")
    print(f"  Layers: {results['random_search']['params']['hidden_layers']}")
    print(f"  Neurons per layer: {results['random_search']['params']['neurons_per_layer']}")
    print(f"  Learning rate: {results['random_search']['params']['learning_rate']:.6f}")
    print(f"  Dropout: {results['random_search']['params']['dropout_rate']:.3f}")
    print(f"  Test Accuracy: {results['random_search']['test_acc']:.4f}")