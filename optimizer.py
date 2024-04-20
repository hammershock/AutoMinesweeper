from itertools import product

import numpy as np
import sympy

import torch
import torch.nn.functional as F
from scipy.optimize import linprog, minimize


# def optimize(weights, bias, num_iter=1000):
#     weights = torch.tensor(weights, dtype=torch.float)
#     bias = torch.tensor(bias, dtype=torch.float)
#
#     n_constraints, x_dim = weights.shape
#     logits = torch.ones(x_dim, requires_grad=True)
#
#     optimizer = torch.optim.Adam([logits], lr=0.01)
#     entropy, penalize = None, None
#
#     for i in range(num_iter):
#         x = F.sigmoid(logits)
#         y_pred = torch.sum(weights * x, dim=1)
#         cost = (y_pred - bias) ** 2
#         entropy = torch.sum(x * torch.log2(x + 10e-5) + (1 - x) * torch.log2(1 - x + 10e-5))
#         penalize = torch.sum(cost) / len(bias)
#         loss = 0.001*entropy + 10000*penalize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if torch.all(cost < 0.01):
#             break
#
#     return x.detach().numpy(), entropy.item(), penalize.item() ** 0.5


def rref(A, b):
    mat = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    rref_matrix, pivot_columns = sympy.Matrix(mat).rref()
    output = np.array(rref_matrix.tolist())
    non_zero_rows = ~np.all(output == 0, axis=1)
    output = output[non_zero_rows]
    return output[:, :-1], output[:, -1].flatten(), pivot_columns[:-1]


def rref_to_variable_representation(w, b, pivot_vars):
    num_vars = w.shape[1]
    free_vars = [i for i in range(num_vars) if i not in pivot_vars]
    num_free_vars = len(free_vars)

    # 初始化权重矩阵和偏置向量
    weight_matrix = np.zeros((num_vars, num_free_vars))
    bias_vector = np.zeros(num_vars)

    for row, bias, pivot_var_index in zip(w, b, pivot_vars):
        weight_matrix[pivot_var_index] = -row[free_vars]
        bias_vector[pivot_var_index] = bias

    for i, free_var_index in enumerate(free_vars):
        weight_matrix[free_var_index, i] = 1

    return weight_matrix, bias_vector


def find_feasible_point_for_linear_boundaries(A, b, epsilon=0):
    # 获取变量的数量
    n_vars = A.shape[1]

    # 目标函数 - 由于我们只需要一个可行点，因此可以设置为零向量
    c = np.zeros(n_vars)

    # 设置上下界，以确保 x 在 0 和 1 之间
    x_bounds = [(0, 1)] * n_vars

    # 扩展 A 和 b 以包含 -A 和 -b 的情况
    A_extended = np.vstack([A, -A])
    b_extended = np.hstack([1 - b - epsilon, b - epsilon])

    # 调用linprog求解线性规划问题
    res = linprog(c, A_ub=A_extended, b_ub=b_extended, bounds=x_bounds, method='highs')

    if res.success:
        # 返回成功找到的内点
        return res.x
    else:
        # 如果没有找到解，则返回一个错误信息
        raise ValueError('No feasible point could be found for the given linear boundaries.')


def distance(x, w, b) -> torch.Tensor:
    dis = torch.square(w@x+b) / torch.sum(torch.square(w), dim=1)
    return dis


def in_bound(x, w, b):
    logits = w@x+b
    return torch.all(logits >= 0) and torch.all(logits <= 1)


def find_solution(w, b):
    available = []
    for x in product(*[(0, 1) for _ in range(w.shape[1])]):
        digits = w @ np.array(x) - b
        print(digits)
        if np.all(digits >= 0) and np.all(digits <= 1):
            available.append(x)
        if len(available) > 10000:
            break
    available = np.array(available)

    return np.mean(available, axis=0), np.log2(len(available))


def optimize(W, b, epsilon=0):
    def objective(x, W, b):
        p = W @ x + b
        # 为了数值稳定性，将p的值限制在[epsilon, 1-epsilon]区间
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1 - epsilon)
        # 计算熵
        H = np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return H

    def objective_jac(x, W, b):
        p = W @ x + b
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1 - epsilon)
        # 计算熵的梯度
        grad_p = np.log2(p) - np.log2(1 - p)
        # 计算x的梯度
        grad_x = W.T @ grad_p
        return grad_x

    def constraint(x, W, b):
        return W @ x + b

    def constraint_jac(x, W, b):
        return W

    x0 = np.random.rand(W.shape[1])

    # 定义约束字典
    cons = [{'type': 'ineq', 'fun': lambda x: 1 - constraint(x, W, b) - epsilon, 'jac': lambda x: -constraint_jac(x, W, b)},
            {'type': 'ineq', 'fun': lambda x: constraint(x, W, b) - epsilon, 'jac': lambda x: constraint_jac(x, W, b)}]

    # 调用minimize函数求解
    res = minimize(objective, x0, args=(W, b), constraints=cons, method='SLSQP')

    if res.success:
        optimal_x = res.x
        optimal_p = W @ optimal_x + b + 10e-6
        entropy = objective(res.x, W, b)
    #     print("Optimal x found:", optimal_x)
    #     print("Resulting p values:", optimal_p)
    #     print("entropy", entropy)
    #     return optimal_p, entropy
    # else:
    #     print("Optimization failed:", res.message)


def _solve(w, b):
    w, b, pivot = rref(w.astype(int), b.astype(int))
    w, b = rref_to_variable_representation(w, b, pivot)
    a = np.all(w == 0, axis=1)
    return optimize(w, b)


if __name__ == "__main__":
    w = torch.tensor([[1, 0], [-1.732, 1], [1.732, 1]], dtype=torch.float)
    b = torch.tensor([0, -1.732, -1.732], dtype=torch.float)

    find_solution(w, b)