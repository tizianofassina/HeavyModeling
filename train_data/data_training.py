import torch
import math
import numpy as np
import random
import math
import yaml
import numpy as np
import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.mixture import GaussianMixture
import scipy.stats as st




def sample_gumbel_copula(size: int, dim: int, theta: float = 3.0) -> np.ndarray:
    """
    Sample from the Gumbel (logistic) copula using NumPy (float64).

    Parameters
    ----------
    size : int
        Number of samples to generate.
    dim : int
        Dimensionality of the multivariate distribution.
    theta : float, optional (default=3.0)
        Copula parameter controlling dependence.

    Returns
    -------
    U : np.ndarray of shape (size, dim)
        Samples from the Gumbel copula.
    """
    a = 1.0 / theta
    U = np.random.rand(size) * np.pi

    W = np.random.exponential(scale=1.0, size=size)

    numerator = np.sin(a * U)
    denominator = (np.sin(U)) ** (1.0 / a)
    factor = (np.sin((1 - a) * U) / W) ** ((1 - a) / a)

    S = numerator / denominator * factor
    S = S.reshape(size, 1)

    E = np.random.exponential(scale=1.0, size=(size, dim))

    U = np.exp(- (E / S) ** a)

    eps = 1e-15
    U = np.clip(U, eps, 1 - eps)

    return U


def log_frechet_rvs_copula_logistic(
    alpha: np.ndarray,
    size: int,
    theta: float = 3.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate multivariate log-Fréchet random samples via the logistic (Gumbel) copula.

    Parameters
    ----------
    alpha : array-like of shape (d,)
        Shape parameters of the log-Fréchet distribution for each dimension.
    size : int
        Number of samples to generate.
    theta : float, optional (default=3.0)
        Logistic copula parameter controlling dependence.
    device : str, optional (default='cpu')
        Target PyTorch device for the returned tensor.

    Returns
    -------
    log_frechet_samples : torch.Tensor of shape (size, d)
        Multivariate log-Fréchet samples as a PyTorch tensor.
    """
    alpha = np.asarray(alpha, dtype=np.float64).reshape(1, -1)

    U = sample_gumbel_copula(size=size, dim=alpha.shape[1], theta=theta)  # float64

    eps = 0  # can be set to 1e-15 for numerical stability if needed
    U = np.clip(U, eps, 1 - eps)

    log_frechet_samples = - (1.0 / alpha) * np.log(-np.log(U))

    log_frechet_samples = torch.tensor(log_frechet_samples, dtype=torch.float32, device=device)

    if not torch.isfinite(log_frechet_samples).all():
        print("Inf or NaN detected")
        print("Number of inf: ", torch.isinf(log_frechet_samples).sum().item())
        print("Number of NaN: ", torch.isnan(log_frechet_samples).sum().item())
        if torch.isnan(log_frechet_samples).any():
            raise ValueError("Numerical error in log-Fréchet samples")

    return log_frechet_samples


def uniform_student_mixture(
        size: int,
        dim: int = 8,
        prob_uniform: float = 0.95,
        gp_params: tuple = (0.0, 1.0, 1.0),
        device: str = "cpu"
) -> torch.Tensor:
    """
    Generate a synthetic dataset with heavy tails and asymmetric, heterogeneous tail dependence.

    With probability `prob_uniform`, samples are drawn from a Uniform(0,1) distribution.
    With probability `1 - prob_uniform`, samples are drawn from independent univariate
    Generalized Pareto (GP) distributions.

    Parameters
    ----------
    size : int
        Number of samples to generate.
    dim : int, optional (default=8)
        Number of dimensions (default is 8 as in the reference setup).
    prob_uniform : float, optional (default=0.95)
        Probability of sampling from Uniform(0,1).
    gp_params : tuple of floats, optional (default=(0.0, 1.0, 1.0))
        Parameters (mu, sigma, xi) of the Generalized Pareto distribution.
        - mu : location
        - sigma : scale
        - xi : shape (tail index)
    device : str, optional (default='cpu')
        Target PyTorch device for the returned tensor.

    Returns
    -------
    samples : torch.Tensor of shape (size, dim)
        Synthetic dataset as a PyTorch tensor.
    """
    mu, sigma, xi = gp_params
    # Decide mixture source for each sample
    mixture_mask = np.random.rand(size) < prob_uniform  # True -> Uniform, False -> GP
    # Initialize array
    data = np.zeros((size, dim), dtype=np.float64)
    # Uniform samples
    uniform_samples = np.random.uniform(low=0.0, high=1.0, size=(mixture_mask.sum(), dim))
    data[mixture_mask] = uniform_samples
    # GP samples (using SciPy's genpareto with c=xi, scale=sigma, loc=mu)
    gp_samples = genpareto.rvs(c=xi, scale=sigma, loc=mu, size=(~mixture_mask).sum() * dim)
    gp_samples = gp_samples.reshape((~mixture_mask).sum(), dim)
    data[~mixture_mask] = gp_samples

    # Convert to PyTorch tensor
    samples = torch.tensor(data, dtype=torch.float32, device=device)

    return samples


def frechet_gaussian_mixture_choice(
        alpha: np.ndarray,
        size: int,
        theta: float = 3.0,
        n_components: int = 2,
        means: np.ndarray = None,
        covariances: np.ndarray = None,
        weights: np.ndarray = None,
        p_frechet: float = 0.5,
        device: str = "cpu"
) -> torch.Tensor:
    """
    Generate samples from a stochastic mixture of a multivariate Fréchet distribution
    (via logistic copula) and a Gaussian Mixture Model (GMM).

    Each sample is drawn from either the Fréchet distribution (with probability p_frechet)
    or the Gaussian Mixture distribution (with probability 1 - p_frechet).

    Parameters
    ----------
    alpha : array-like of shape (d,)
        Shape parameters of the Fréchet distribution for each dimension.
    size : int
        Number of samples to generate.
    theta : float, optional (default=3.0)
        Logistic copula parameter for the Fréchet part.
    n_components : int, optional (default=2)
        Number of Gaussian components in the mixture.
    means : np.ndarray of shape (n_components, d), optional
        Means of the Gaussian components. If None, random initialization is used.
    covariances : np.ndarray of shape (n_components, d, d), optional
        Covariance matrices for the components. If None, identity matrices are used.
    weights : np.ndarray of shape (n_components,), optional
        Mixture weights for the Gaussian components. If None, uniform weights are used.
    p_frechet : float, optional (default=0.5)
        Probability of sampling from the Fréchet distribution.
    device : str, optional (default='cpu')
        Target PyTorch device for the returned tensor.

    Returns
    -------
    samples : torch.Tensor of shape (size, d)
        Mixture samples as a PyTorch tensor.
    """
    d = len(alpha)

    # Step 1: Fréchet samples (exp of log-Fréchet)
    log_frechet_samples = log_frechet_rvs_copula_logistic(
        alpha, size, theta=theta, device="cpu"
    ).cpu().numpy().astype(np.float64)

    frechet_samples = torch.tensor(
        np.exp(log_frechet_samples),
        device=device,
        dtype=torch.float32
    )

    # Step 2: Gaussian Mixture setup
    if means is None:
        means = np.random.randn(n_components, d)
    if covariances is None:
        covariances = np.array([np.eye(d) for _ in range(n_components)])
    if weights is None:
        weights = np.ones(n_components) / n_components

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full"
    )
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

    gmm_samples, _ = gmm.sample(size)
    gmm_samples = torch.tensor(gmm_samples, dtype=torch.float32, device=device)

    # Step 3: Bernoulli selection
    bernoulli_mask = np.random.rand(size) < p_frechet
    bernoulli_mask = torch.tensor(bernoulli_mask, dtype=torch.bool, device=device)

    samples = torch.zeros_like(frechet_samples, device=device)
    samples[bernoulli_mask] = frechet_samples[bernoulli_mask]
    samples[~bernoulli_mask] = gmm_samples[~bernoulli_mask]

    # Step 4: Check for NaN and Inf
    num_nan = torch.isnan(samples).sum().item()
    num_pos_inf = torch.isinf(samples).logical_and(samples > 0).sum().item()
    num_neg_inf = torch.isinf(samples).logical_and(samples < 0).sum().item()

    if num_nan > 0 or num_pos_inf > 0 or num_neg_inf > 0:
        print(f"NaN count: {num_nan}")
        print(f"+Inf count: {num_pos_inf}")
        print(f"-Inf count: {num_neg_inf}")
        if num_pos_inf > 0 or num_neg_inf > 0:
            raise ValueError("Infinite values detected in final samples.")

    return samples


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_means(dim):
    """
    Generate a set of mode centers on a 2D grid, then repeat coordinates to reach the desired dimension.

    Parameters
    ----------
    dim : int
        Target dimensionality of each mode (must be even).

    Returns
    -------
    grid : np.ndarray of shape (25, dim)
        Array of mode centers.
    """
    coords = np.array([-2, -1, 0, 1, 2], dtype=np.float32) * 5
    xx, yy = np.meshgrid(coords, coords, indexing='ij')
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    # Repeat along columns to reach dim
    grid = np.tile(grid, (1, dim // 2)).astype(np.float32)
    return grid

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device : ", device)





    alpha = [2., 3.]


    # --- GENERATING DATA ---

    set_seed(42)
    dim = len(alpha)
    sample = st.chi2(df=1).rvs(size=25)
    weights = np.array(sample / np.sum(sample))
    means = generate_means(dim)


    train_mix_gaussian = frechet_gaussian_mixture_choice(alpha=alpha, size=100_000,
                                                         theta=3.0, n_components=25, means=means, covariances=None,
                                                         weights=weights, p_frechet=0.5, device=device)


    test_mix_gaussian = frechet_gaussian_mixture_choice(alpha=alpha, size=30_000_000,
                                                        theta=3.0, n_components=25, means=means, covariances=None,
                                                        weights=weights, p_frechet=0.5, device=device)

    set_seed(42)
    train_log_frechet = log_frechet_rvs_copula_logistic(alpha = alpha, size = 100_000, theta=3.0, device=device)
    test_log_frechet = log_frechet_rvs_copula_logistic(alpha = alpha, size = 30_000_000, theta=3.0, device=device)

    train_log_frechet = log_frechet_rvs_copula_logistic(alpha=alpha, size=100_000,
                                                        theta=3.0, device=device)
    train_big_log_frechet = log_frechet_rvs_copula_logistic(alpha=alpha, size=1_000_000, theta=3.0, device=device)
    test_log_frechet = log_frechet_rvs_copula_logistic(alpha=alpha, size=30_000_000, theta=3.0,
                                                       device=device)

    set_seed(42)
    train_uniform = uniform_student_mixture(size = 100_000, dim = 2, prob_uniform = 0.95, gp_params = (0.0, 1.0, 1.0), device = device)
    test_uniform = uniform_student_mixture(size = 30_000_000, dim = 2, prob_uniform = 0.95, gp_params = (0.0, 1.0, 1.0), device = device)


    # --- SAVING DATA ---
    torch.save(train_log_frechet, "training_frechet_10e5.pt")
    torch.save(train_big_log_frechet, "training_frechet_10e6.pt")
    torch.save(test_log_frechet, "test_frechet.pt")

    torch.save(train_uniform, "training_uniform.pt")
    torch.save(test_uniform, "test_uniform.pt")

    torch.save(train_mix_gaussian, "training_mix_gaussian.pt")
    torch.save(test_mix_gaussian, "test_mix_gaussian.pt")






