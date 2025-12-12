import torch
from tqdm import tqdm

class GWOOptimizer:
    """
    Grey Wolf Optimizer (GWO) implementation in PyTorch.
    Fully GPU-compatible, supports per-dimension bounds.
    """

    def __init__(self, objective_function, bounds, population_size=6, max_iter=1000, device=None):
        """
        Args:
            objective_function: function to minimize; should take a 1D tensor input.
            bounds: list of tuples [(low1, high1), (low2, high2), ...]
            population_size: number of wolves
            max_iter: maximum number of iterations
            device: 'cpu' or 'cuda'; defaults to auto-detect
        """
        self.objective_function = objective_function
        self.population_size = population_size
        self.max_iter = max_iter
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Dimension
        self.dim = len(bounds)

        # Bounds as tensors
        self.lowers = torch.tensor([b[0] for b in bounds], dtype=torch.float32, device=self.device)
        self.uppers = torch.tensor([b[1] for b in bounds], dtype=torch.float32, device=self.device)

        # Exploration parameter schedule
        self.a_range = torch.linspace(2.0, 0.001, max_iter, device=self.device)

        # Initialize population
        self.agents = self._initialize_population()

        # Leaders
        self.alpha = None
        self.beta = None
        self.delta = None
        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

    def _initialize_population(self):
        """Randomly initialize population within bounds."""
        rand = torch.rand((self.population_size, self.dim), device=self.device)
        population = self.lowers + rand * (self.uppers - self.lowers)
        return population

    def _clamp(self, positions):
        """Clamp positions to stay within bounds."""
        return torch.min(torch.max(positions, self.lowers), self.uppers)

    def _evaluate_population(self):
        """Safely evaluate all agents, ensuring proper tensor shape."""
        # Evaluate each agent individually
        scores = []
        for agent in self.agents:
            score = self.objective_function(agent)
            # Ensure score is a scalar
            if isinstance(score, torch.Tensor):
                score = score.item() if score.dim() == 0 else score.squeeze().item()
            scores.append(score)
        
        # Convert to tensor with proper shape
        return torch.tensor(scores, device=self.device)

    def optimize(self):
        """Run the optimization."""
        loss_curve = []

        for i in tqdm(range(self.max_iter)):
            # Evaluate all agents
            Z = self._evaluate_population()  # Shape: [population_size]
            
            # Sort and select alpha, beta, delta
            Z_sorted, idx = torch.sort(Z)
            
            # Ensure we have at least 3 agents for alpha, beta, delta
            if self.population_size >= 3:
                self.alpha = self.agents[idx[0]]
                self.beta = self.agents[idx[1]] 
                self.delta = self.agents[idx[2]]
                self.alpha_score = Z_sorted[0].item()
                self.beta_score = Z_sorted[1].item()
                self.delta_score = Z_sorted[2].item()
            else:
                # Handle small population sizes
                self.alpha = self.agents[idx[0]]
                self.alpha_score = Z_sorted[0].item()
                if self.population_size >= 2:
                    self.beta = self.agents[idx[1]]
                    self.beta_score = Z_sorted[1].item()
                if self.population_size >= 3:
                    self.delta = self.agents[idx[2]]
                    self.delta_score = Z_sorted[2].item()
            
            loss_curve.append(self.alpha_score)

            a = self.a_range[i]

            # Random coefficients
            A1 = 2 * a * torch.rand_like(self.agents) - a
            A2 = 2 * a * torch.rand_like(self.agents) - a
            A3 = 2 * a * torch.rand_like(self.agents) - a

            C1 = 2 * torch.rand_like(self.agents)
            C2 = 2 * torch.rand_like(self.agents)
            C3 = 2 * torch.rand_like(self.agents)

            # Distances - handle cases where beta/delta might not exist
            D_alpha = torch.abs(C1 * self.alpha - self.agents)
            
            if self.beta is not None:
                D_beta = torch.abs(C2 * self.beta - self.agents)
                X2 = self.beta - A2 * D_beta
            else:
                X2 = self.alpha.clone()  # Fall back to alpha
                
            if self.delta is not None:
                D_delta = torch.abs(C3 * self.delta - self.agents)
                X3 = self.delta - A3 * D_delta
            else:
                X3 = self.alpha.clone()  # Fall back to alpha

            # Update positions
            X1 = self.alpha - A1 * D_alpha
            self.agents = (X1 + X2 + X3) / 3.0

            # Clamp to bounds
            self.agents = self._clamp(self.agents)

        return self.alpha, loss_curve