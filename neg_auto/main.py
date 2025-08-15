import torch
from torchjd import mtl_backward
from torch.nn import Linear, ReLU, Sequential
from torchjd.aggregation import UPGrad
from torch import nn
from torch.optim import SGD
from scipy.optimize import fsolve
import numpy as np
from tqdm import tqdm
from eq import S_alpha_xss_analytic, S_n_xss_analytic, generate_initial_guesses, Equs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

t = 0

def ssfinder(alpha_val,n_val):

    InitGuesses = generate_initial_guesses(alpha_val, n_val)

    params = torch.tensor([alpha_val, n_val])

    for InitGuess in InitGuesses:

        output, infodict, intflag, _ = fsolve(Equs, InitGuess, args=(t, params), xtol=1e-12, full_output=True)
        xss = torch.from_numpy(output)
        fvec = torch.from_numpy(infodict['fvec'])

        delta = 1e-8
        dEqudx = (Equs(xss+delta, t, params)-Equs(xss, t, params))/delta
        jac = torch.tensor([[dEqudx]])
        eig = jac
        instablility = torch.real(eig) >= 0


        if xss > 0.04 and torch.linalg.norm(fvec) < 1e-10 and intflag == 1 and instablility==False:
            return xss[0]

    return float('nan')

def objective_function_batch(alpha_vec, n_vec):
    batch_size = alpha_vec.shape[0]
    S_alpha_list = []
    S_n_list = []
    alpha_list = []
    n_list = []
    for i in range(batch_size):
        alpha_i = alpha_vec[i]
        n_i = n_vec[i]
        xss = torch.tensor(ssfinder(alpha_i.item(), n_i.item()))
        if np.isnan(xss):
            continue
        else:
            S_alpha_val = S_alpha_xss_analytic(xss, alpha_i, n_i)
            S_n_val = S_n_xss_analytic(xss, alpha_i, n_i)
            S_alpha_list.append(torch.tensor(S_alpha_val, requires_grad=True, device=device))
            S_n_list.append(torch.tensor(S_n_val, requires_grad=True, device=device))
            alpha_list.append(torch.tensor(alpha_i, requires_grad=True, device=device))
            n_list.append(torch.tensor(n_i, requires_grad=True, device=device))
    return torch.stack(S_alpha_list), torch.stack(S_n_list)

shared_module = Sequential(Linear(2, 4), ReLU(), Linear(4, 8), ReLU())
alpha_module = Linear(8, 1)
n_module = Linear(8, 1)
params = [
    *shared_module.parameters(),
    *alpha_module.parameters(),
    *n_module.parameters(),
]

optimizer = SGD(params, lr=0.1)
aggregator = UPGrad()

# pareto_front = []

batch_size = 64
num_epochs = 100

with tqdm(total=num_epochs, desc="Optimizing Population", ncols=100) as pbar:
    for epoch in range(num_epochs):
        alpha = nn.Parameter(torch.FloatTensor(batch_size).uniform_(0.01, 50).to(device), requires_grad=True)
        n = nn.Parameter(torch.FloatTensor(batch_size).uniform_(0.01, 10).to(device), requires_grad=True)
        S_alpha, S_n, alpha_list, n_list = objective_function_batch(alpha, n)

        S_alpha.retain_grad()
        S_n.retain_grad()
        alpha_list.retain_grad()
        n_list.retain_grad()

        input = torch.stack([alpha_list, n_list], dim=1)

        features = shared_module(input)
        output1 = alpha_module(features)
        output2 = n_module(features)

        optimizer.zero_grad()

        mtl_backward(losses=[S_alpha, S_n], features=features, aggregator=aggregator)

        optimizer.step()

        # valid_mask = torch.isfinite(S_alpha) & torch.isfinite(S_n)
        # pareto_front.extend(torch.stack([S_alpha[valid_mask], S_n[valid_mask]], dim=1).detach().cpu().numpy())

        pbar.update(1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Median Loss = {((S_alpha + S_n)/2).median().item():.4f}")

# pareto_front_np = np.array(pareto_front)

# np.savetxt("points.csv", pareto_front_np, delimiter=",", header="x,y", comments="")