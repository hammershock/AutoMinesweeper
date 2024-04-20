from collections import defaultdict
from itertools import product

import numpy as np

from Board import Board
from optimizer import optimize, _solve, rref_to_variable_representation, rref, find_solution
from utils import pad


class Solver:
    def __init__(self, board: Board):
        self.results = None
        self.best = None
        self.board = board

    def observe(self):
        image = self.board.render()
        mask_unclicked = image >= 10
        mask_edge = np.zeros_like(mask_unclicked, dtype=bool)

        for i, j in np.argwhere(np.bitwise_and(image < 10, image >= 0)):
            mask_edge[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True

        mask_left = np.bitwise_and(mask_unclicked, ~mask_edge)
        unknown_pos2id = {tuple(item): i for i, item in enumerate(np.argwhere(np.bitwise_and(mask_unclicked, mask_edge)))}

        equations = defaultdict(list)
        for i, j in np.argwhere(np.bitwise_and(image < 10, image > 0)):
            dd = [-1, 0, 1]
            for dx, dy in product(dd, dd):
                if (i + dx, j + dy) in unknown_pos2id:
                    equations[(i, j)].append((i + dx, j + dy))

        equations_arr = np.zeros((len(equations), len(unknown_pos2id)), dtype=int)
        equations_sum = np.zeros(len(equations), dtype=int)

        for i, (digit_place, values) in enumerate(equations.items()):
            equations_sum[i] = image[digit_place]
            for value in values:
                equations_arr[i, unknown_pos2id[value]] += 1
        return equations_arr, equations_sum, unknown_pos2id, mask_left

    def i_solve(self):
        t, r, unknown_pos2id, left = self.observe()
        if t.size > 0:
            mark0 = np.zeros_like(t[0])
            mark1 = np.zeros_like(t[0])
            last_count = 0
            while True:
                for i, row in enumerate(t):
                    r[i] -= np.sum(row * mark1)
                    row[mark1 > 0] = 0
                    row[mark0 > 0] = 0

                    if r[i] == 0:
                        mark0[row > 0] = 1
                    elif np.sum(row) == r[i]:
                        mark1[row > 0] = 1

                count = np.sum(mark1) + np.sum(mark0)
                if count == last_count:
                    break
                last_count = count

            p_array = np.full_like(mark0, fill_value=-1, dtype=float)
            p_array[mark0 > 0] = 0
            p_array[mark1 > 0] = 1
            uncertain_mask = p_array < 0
            # mines_remain_count = self.board.num_landmine - np.sum(mark1)
            # print(len(mark0), np.sum(p_array >= 0))
            T = []
            R = []
            for rr, row in zip(r, t):
                if np.sum(row > 0):
                    T.append(row[uncertain_mask].tolist())
                    R.append(rr)

            results = {}
            if len(T):
                T = np.array(T)
                R = np.array(R)
                # w, b, pivot = rref(T.astype(int), R.astype(int))
                # w, b = rref_to_variable_representation(w, b, pivot)

                res = find_solution(T, R)

                if res is not None:
                    p, entropy = res
                    print(f"熵:{entropy}")
                    p_array[uncertain_mask] = p

                results.update({place: p_array[i] for place, i in unknown_pos2id.items()})
                self.best = min((prob, place) for place, prob in results.items())[-1]
                self.results = results

