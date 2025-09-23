import numpy as np
import torch
from tqdm import tqdm


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


class FittingMonitor:

    def __init__(self, ftol=1e-5, gtol=1e-6, maxiters=100, verbose=False):
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def run_fitting(self, optimizer, closure, params):
        prev_loss = None
        grad_require(params, True)
        if self.verbose:
            trange = tqdm(range(self.maxiters), desc='Fitting')
        else:
            trange = range(self.maxiters)
        print('maxiters', self.maxiters)
        for iter in trange:
            print('iter', iter)
            loss = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
            #         for var in params if var.grad is not None]):
            #     print('Small grad, stopping!')
            #     break

            if iter > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                # print(loss_rel_change, self.ftol)

                if loss_rel_change <= self.ftol:
                    break

            prev_loss = loss.item()
        grad_require(params, False)
        return prev_loss

    def close(self):
        pass


def grad_require(paras, flag=False):
    if isinstance(paras, list):
        for par in paras:
            par.requires_grad = flag
    elif isinstance(paras, dict):
        for key, par in paras.items():
            par.requires_grad = flag
