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

    def optimize(self):
        """Run the optimization."""
        loss_curve = []

        for i in tqdm(range(self.max_iter)):
            # Evaluate all agents
            Z = torch.tensor([self.objective_function(x) for x in self.agents], device=self.device)

            # Sort and select alpha, beta, delta
            Z_sorted, idx = torch.sort(Z)
            self.alpha, self.beta, self.delta = self.agents[idx[0]], self.agents[idx[1]], self.agents[idx[2]]
            loss_curve.append(Z_sorted[0].item())

            a = self.a_range[i]

            # Random coefficients
            A1 = 2 * a * torch.rand_like(self.agents) - a
            A2 = 2 * a * torch.rand_like(self.agents) - a
            A3 = 2 * a * torch.rand_like(self.agents) - a

            C1 = 2 * torch.rand_like(self.agents)
            C2 = 2 * torch.rand_like(self.agents)
            C3 = 2 * torch.rand_like(self.agents)

            # Distances
            D_alpha = torch.abs(C1 * self.alpha - self.agents)
            D_beta  = torch.abs(C2 * self.beta  - self.agents)
            D_delta = torch.abs(C3 * self.delta - self.agents)

            # Update positions
            X1 = self.alpha - A1 * D_alpha
            X2 = self.beta  - A2 * D_beta
            X3 = self.delta - A3 * D_delta
            self.agents = (X1 + X2 + X3) / 3.0

            # Clamp to bounds
            self.agents = self._clamp(self.agents)

        return self.alpha, loss_curve
