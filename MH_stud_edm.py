import torch
import numpy as np
from ot.sliced import sliced_wasserstein_distance
from heavytail_tools.general_tools import set_seed
import torch
from torch.distributions import StudentT, Chi2
import time


def log_frechet_multivariate(x, alpha, tmp=None):
    alpha = alpha.view(1, -1)
    if tmp is None:
        tmp = x.new_empty(x.shape)
    tmp.copy_(x)
    tmp.pow_(-alpha)
    logp = torch.sum(torch.log(alpha) - (alpha + 1) * torch.log(x) - tmp, dim=1)
    return logp

def log_gaussian(x, y, sigma, tmp=None):
    if tmp is None:
        tmp = x.new_empty(x.shape)
    tmp.copy_(y)
    tmp.sub_(x)
    tmp.pow_(2)
    logp = -torch.sum(tmp, dim=1) / (2 * sigma**2)
    return logp

def log_prob(x, y, alpha, sigma, tmp_frechet=None, tmp_gauss=None):
    return log_frechet_multivariate(x, alpha, tmp=tmp_frechet) + log_gaussian(x, y, sigma, tmp=tmp_gauss)

def mh_sampler_x0_given_xt_student(
    x0_init, x_t, alpha, sigma,
    n_steps=1000, proposal_scale=0.5, df=5
):
    device = x0_init.device
    n, d = x0_init.shape
    x = x0_init.clone().to(device).clamp(min=1e-12)

    Z = torch.empty_like(x)
    V = torch.empty_like(x)
    t_noise = torch.empty_like(x)
    chi_samples = torch.empty((n, df, d), device=device)
    tmp_frechet = torch.empty_like(x)
    tmp_gauss = torch.empty_like(x)

    for i in range(n_steps):
        start_total = time.time()

        Z.normal_()
        chi_samples.normal_()
        chi_samples.pow_(2)
        V.copy_(chi_samples.sum(dim=1))

        t_noise.copy_(Z / torch.sqrt(V / df))
        x_prop = x + proposal_scale * t_noise

        logp_current = log_prob(x, x_t, alpha, sigma, tmp_frechet, tmp_gauss)
        logp_prop = log_prob(x_prop, x_t, alpha, sigma, tmp_frechet, tmp_gauss)
        log_accept = logp_prop - logp_current

        accept = (torch.rand_like(x[:, 0]) < torch.exp(log_accept)).unsqueeze(1).float()
        x.mul_(1 - accept).add_(accept * x_prop)

        #if i % 10 == 0:
        #    print(f"Step {i}: total_time = {time.time() - start_total:.6f}s")

    return x

def expectancy_MC_mh(
    x0_init, x_t, sigma, alpha,
    n_steps=1000, proposal_scale=0.5,
    batch_size=100, n_MC=100
):
    device = x_t.device
    x0_init = x0_init.to(device, non_blocking=True)
    alpha = alpha.to(device, non_blocking=True)

    x = mh_sampler_x0_given_xt_student(
        x0_init=x0_init,
        x_t=x_t,
        alpha=alpha,
        sigma=sigma,
        n_steps=n_steps,
        proposal_scale=proposal_scale
    )

    x = x.view(batch_size, n_MC, -1)
    MC = x.mean(dim=1)
    
    return MC

def generator_edm_mh(
    samples,
    n_steps_edm=20,
    sigma_min=0.002,
    sigma_max=7,
    rho=7.,
    device="cpu",
    alpha=torch.tensor([3., 4.]),
    n_MC=100,
    proposal_scale=0.5,
    n_steps_mh=1000
):
    samples = samples.to(device, non_blocking=True)
    alpha = alpha.to(device, non_blocking=True)
    batch_size, d = samples.shape

    t = torch.linspace(0, 1, n_steps_edm, device=device)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + t * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho

    x = samples.clone()

    with torch.no_grad():
        expanded_shape = (batch_size * n_MC, d)
        for i in range(len(sigmas) - 1):
            sigma_t = sigmas[i].item()
            sigma_s = sigmas[i + 1].item()

            U = torch.rand(expanded_shape, device=device)
            x0_init = (-torch.log(U)) ** (-1 / alpha)

            x_reshape = x.unsqueeze(1).expand(-1, n_MC, -1).reshape(expanded_shape)

            x0_pred = expectancy_MC_mh(
                x0_init=x0_init,
                x_t=x_reshape,
                sigma=sigma_t,
                alpha=alpha,
                n_steps=n_steps_mh,
                proposal_scale=proposal_scale,
                batch_size=batch_size,
                n_MC=n_MC
            )

            k1 = (x - x0_pred) / sigma_t
            x_pred = x + (sigma_s - sigma_t) * k1

            x_pred_reshape = x_pred.unsqueeze(1).expand(-1, n_MC, -1).reshape(expanded_shape)

            x0_pred_pred = expectancy_MC_mh(
                x0_init=x0_init,
                x_t=x_pred_reshape,
                sigma=sigma_s,
                alpha=alpha,
                n_steps=n_steps_mh,
                proposal_scale=proposal_scale,
                batch_size=batch_size,
                n_MC=n_MC
            )

            k2 = (x_pred - x0_pred_pred) / sigma_s
            x.add_((sigma_s - sigma_t) * 0.5 * (k1 + k2))

            #if i % 20 == 0:
            print("Step:", i, "out of", n_steps_edm)

    return x



if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    set_seed(100)
    print("Testing EDM with MH sampling")

    batch_size = 100_000
    d = 1
    n_MC = 30
    n_steps_mh = 3000
    proposal_scale = 0.5
    n_steps_edm = 100
    sigma_min = 0.002
    sigma_max = 7.0
    rho = 7.0
    alpha = torch.tensor([5.], device=device)

    U = torch.rand((batch_size, d), device=device)
    frechet = (-torch.log(U))**(-1/alpha)
    samples = sigma_max * torch.randn((batch_size, d), device=device) + frechet
    print("Initial samples mean:", samples.mean(dim=0))
    x_generated = generator_edm_mh(
        samples,
        n_steps_edm=n_steps_edm,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        alpha=alpha,
        n_MC=n_MC,
        proposal_scale=proposal_scale,
        n_steps_mh=n_steps_mh
    )

    print("Shape:", x_generated.shape)
    print("Media campioni :", x_generated.mean(dim=0))
    print("Std campioni :", x_generated.std(dim=0))

    # Riproduzione Frechet per confronto
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
    print("Real Wass : ", sliced_wasserstein_distance(frechet_3.cpu().numpy(), frechet_4.cpu().numpy(), n_projections=10, p=2))
    print("Gen Wass : ", sliced_wasserstein_distance(x_generated.cpu().numpy(), frechet_4.cpu().numpy(), n_projections=10, p=2))