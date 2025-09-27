import torch


import numpy as np
from ot.sliced import sliced_wasserstein_distance

from heavytail_tools.general_tools import check_non_finite_torch, set_seed




def log_frechet_multivariate(x, alpha):
    alpha = alpha.view(1, -1)
    logp = torch.sum(torch.log(alpha) - (alpha + 1) * torch.log(x) - x**(-alpha), dim=1)
    return logp


def log_gaussian(x,y, sigma):
    return -torch.sum((y - x)**2, dim=1) / (2 * sigma**2)

def log_prob(x,y, alpha, sigma):
    return log_frechet_multivariate(x, alpha) + log_gaussian(x, y, sigma)


def log_frechet_multivariate_grad(x, alpha):
    alpha = alpha.view(1, -1).expand_as(x)
    x_safe = x.clamp(min=1e-12)
    grad = -(alpha + 1)/x_safe + alpha * x_safe**(-alpha - 1)
    grad = torch.clamp(grad, min=-1e12, max=1e12)
    return grad


def gaussian_score(x_0, x_t, sigma):
    grad = (x_t - x_0) / (sigma**2)
    return  grad


def score_for_mala(x_0, x_t, alpha, sigma):
    score_frechet = log_frechet_multivariate_grad(x_0, alpha)
    score_gaussian = gaussian_score(x_0, x_t, sigma)
    return score_frechet + score_gaussian


import torch

def mala_sampler_x0_given_xt(x0_init, x_t, alpha, sigma, n_steps=1000, step_size=0.1):
    n, d = x0_init.shape
    x = x0_init.clone().clamp(min=1e-4)
    device = x0_init.device
    for _ in range(n_steps):
        grad = score_for_mala(x, x_t, alpha, sigma)

        x_prop = x + 0.5 * step_size**2 * grad + step_size * torch.randn_like(x, device= device)
        x_prop = torch.clamp(x_prop, min=1e-6)

        logp_current = log_prob(x,x_t, alpha, sigma)
        logp_prop = log_prob(x_prop, x_t, alpha, sigma)

        grad_prop = score_for_mala(x_prop, x_t, alpha, sigma)

        logq_forward = -torch.sum((x_prop - x - 0.5 * step_size**2 * grad)**2, dim=1) / (2 * step_size**2)
        logq_backward = -torch.sum((x - x_prop - 0.5 * step_size**2 * grad_prop)**2, dim=1) / (2 * step_size**2)
        log_accept = logp_prop - logp_current + logq_backward - logq_forward

        accept = (torch.rand(n, device = device) < torch.exp(log_accept)).float().view(-1, 1)
        x = accept * x_prop + (1 - accept) * x
    return x  # shape (n, d)


def expectancy_MC(x0_init, x_t, sigma, alpha, n_steps=1000, step_size=0.1, batch_size=100, n_MC=100):
    device = x_t.device
    x0_init = x0_init.to(device)
    alpha = alpha.to(device)

    x = mala_sampler_x0_given_xt(
        x0_init=x0_init,
        x_t= x_t,
        alpha=alpha,
        sigma=sigma,
        n_steps=n_steps,
        step_size=step_size
    )

    x = x.reshape((batch_size, n_MC, x.shape[-1])).to(device)
    MC = x.mean(dim=1)
    return MC

def generator_edm(
        samples,
        n_steps_edm=20,
        sigma_min=0.002,
        sigma_max=7,
        rho=7.,
        device="cpu",
        alpha=torch.tensor([3., 4.]),
        n_MC=100,
        step_size=0.1,
        n_steps_mala=1000
):
    samples = samples.to(device)
    alpha = alpha.to(device)

    batch_size = samples.shape[0]
    d = samples.shape[1]

    t = torch.linspace(0, 1, n_steps_edm, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

    x = samples.clone().to(device)

    with torch.no_grad():
        for i in range(len(sigmas) - 1):
            sigma_t = sigmas[i].item()
            sigma_s = sigmas[i + 1].item()

            U = torch.rand((batch_size * n_MC, d), device=device)

            x_reshape = x.unsqueeze(1).repeat(1, n_MC, 1).reshape(batch_size * n_MC, d).to(device)

            x0_init = (-torch.log(U)) ** (-1 / alpha)

            x0_pred = expectancy_MC(
                x0_init=x0_init.to(device),
                x_t=x_reshape,
                sigma=sigma_t,
                alpha=alpha,
                n_steps=n_steps_mala,
                step_size=step_size,
                batch_size=batch_size,
                n_MC=n_MC
            )

            k1 = (x - x0_pred) / sigma_t

            x_pred = x + (sigma_s - sigma_t) * k1
            x_pred_reshape = x_pred.unsqueeze(1).repeat(1, n_MC, 1).reshape(batch_size * n_MC, d).to(device)

            x0_pred_pred = expectancy_MC(
                x0_init=x0_init.to(device),
                x_t=x_pred_reshape,
                sigma=sigma_s,
                alpha=alpha,
                n_steps=n_steps_mala,
                step_size=step_size,
                batch_size=batch_size,
                n_MC=n_MC
            )

            k2 = (x_pred - x0_pred_pred) / sigma_s

            x = x + (sigma_s - sigma_t) * 0.5 * (k1 + k2)
            print("Step : ", i, " out of ", n_steps_edm)

    return x
if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    set_seed(100)
    batch_size = 1_0_000
    d = 1
    n_MC = 20
    n_steps_mala = 2000
    step_size = 0.1
    n_steps_edm = 100
    sigma_min = 0.002
    sigma_max = 7.0
    rho = 7.0
    alpha = torch.tensor([5.], device = device)

    U = torch.rand((batch_size, d), device=device)
    frechet = (-torch.log(U))**(-1/alpha)
    samples = sigma_max*torch.randn((batch_size, d), device=device) + frechet

    x_generated = generator_edm(
        samples,
        n_steps_edm=n_steps_edm,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        alpha=alpha,
        n_MC=n_MC,
        step_size=step_size,
        n_steps_mala=n_steps_mala
    )

    print("Shape:", x_generated.shape)
    print("Media campioni : ", x_generated.mean(dim=0))
    print("Std campioni : ", x_generated.std(dim=0))

    U = torch.rand((batch_size, d), device=device)
    frechet = (-torch.log(U)) ** (-1 / alpha)
    U = torch.rand((batch_size, d), device=device)
    frechet_2 = (-torch.log(U)) ** (-1 / alpha)
    U = torch.rand((batch_size, d), device=device)
    frechet_3 = (-torch.log(U)) ** (-1 / alpha)
    U = torch.rand((batch_size, d), device=device)
    frechet_4 = (-torch.log(U)) ** (-1 / alpha)

    q = np.array([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])

    quan_1 = np.quantile(frechet[:, 0].cpu().numpy(), q)
    quan_2 = np.quantile(frechet_2[:, 0].cpu().numpy(), q)
    quan_gen = np.quantile(x_generated[:, 0].cpu().numpy(), q)
    print("Frechet_1 quantile", quan_1)
    print("Frechet_2 quantile", quan_2)
    print("Frechet_gen quantile", quan_gen)

    print("Real Wass : ", sliced_wasserstein_distance(frechet.cpu().numpy(), frechet_2.cpu().numpy(), n_projections=10, p=2))
    print("Real Wass : ", sliced_wasserstein_distance(frechet.cpu().numpy(), frechet_2.cpu().numpy(), n_projections=10, p=2))
    print("Gen Wass : ", sliced_wasserstein_distance(frechet_3.cpu().numpy(), frechet_4.cpu().numpy(), n_projections=10, p=2))
