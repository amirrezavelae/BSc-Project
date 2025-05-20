import gym
import numpy as np

from .model_saver import ModelSaver
from .plotting import Plotter
from .replay_buffer import ReplayBuffer
from .trainer import train

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rewards_to_go(rewards, discounting=0.99):
    res = torch.zeros_like(rewards)
    last = 0.
    for i in reversed(range(rewards.shape[0])):
        last = res[i][0] = rewards[i][0] + discounting * last

    return res


def running_average(arr, smoothing=0.8):
    size = len(arr)
    res = np.zeros(size)

    if size == 0:
        return res

    res[0] = arr[0]
    for i in range(1, size):
        res[i] = res[i - 1] * smoothing + arr[i] * (1 - smoothing)

    return res


def one_hot(x, size):
    res = torch.zeros(size)
    res[x] = 1
    return res


def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.flatten()
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def bootstrap(rewards, last, discounting=0.99):
    res = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last = res[i] = rewards[i] + discounting * last
    return res


def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()


# Conjugate gradient algorithm (accepts tensors)
def conjugate_gradient(A, b, delta=0., max_iterations=float('inf'), print_iterations=False):
    x = torch.zeros_like(b)
    # x_i = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new

        # if i == 10:
        #     # First iteration, set x_i to the initial guess
        #     x_i = x.clone()
    # if print_iterations:
    #     print(
    #         f"Iterations: {i} and norm diff: {torch.norm(x - x_i).item():.3f} and norm x: {torch.norm(x).item():.3f}")
    return x


def bicg(A, AT, b, delta=1e-6, max_iterations=1000, print_iterations=False):
    """
    Bi-Conjugate Gradient algorithm to solve Ax = b

    Args:
        A: A function that computes A @ v
        AT: A function that computes A.T @ v
        b: Right-hand side vector
        delta: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Approximate solution x ≈ A⁻¹b
    """
    x = torch.zeros_like(b)
    r = b - A(x)
    r_bar = r.clone()
    p = r.clone()
    p_bar = r_bar.clone()

    k = 0
    while k < max_iterations:
        Ap = A(p)
        ATp_bar = AT(p_bar)
        denom = (p_bar @ Ap)
        if denom.abs() < 1e-10:
            break  # Avoid division by zero or numerical instability

        alpha = (r_bar @ r) / denom

        x_new = x + alpha * p
        if (x_new - x).norm() < delta:
            return x_new

        r_new = r - alpha * Ap
        r_bar_new = r_bar - alpha * ATp_bar

        beta = (r_bar_new @ r_new) / (r_bar @ r)

        p = r_new + beta * p
        p_bar = r_bar_new + beta * p_bar

        # Prepare for next iteration
        x = x_new
        r = r_new
        r_bar = r_bar_new
        k += 1
    if print_iterations:
        print(k)
    return x
