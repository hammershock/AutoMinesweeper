from itertools import product

import numpy as np

from utils import pad, generate_board


class BaseBoard:
    blank = 0
    landmine = -1
    landmine_clicked = -2

    mark_unknown = 10
    mark_flag = 11
    mark_question = 12
    mark_edge = 13

    STATE_GAMING = 0
    STATE_WIN = 1
    STATE_LOSE = 2

    def __init__(self, num_landmine, shape: tuple = (9, 9), callback_fn=None):
        self._board = np.zeros(shape, dtype=int)
        self._visibility = np.zeros(shape, dtype=bool)
        self._marks = np.zeros(shape, dtype=int)

        self.state = self.STATE_GAMING
        self.finished = False
        self.image = np.full(shape, fill_value=self.mark_unknown, dtype=int)
        self.num_landmine = num_landmine
        self.rest_count = num_landmine
        self.callback_fn = callback_fn

    def _check_win(self) -> None:
        if np.argwhere(self._marks == 1).tobytes() == np.argwhere(self._board < 0).tobytes() == np.argwhere(
                ~self._visibility).tobytes():
            self._win()

    def _win(self) -> None:
        self.finished = True
        self.state = self.STATE_WIN

    def _lose(self) -> None:
        self.finished = True
        self.state = self.STATE_LOSE
        self._visibility = np.ones_like(self._visibility)

    def _flood_fill(self, i: int, j: int, padded_board: np.ndarray, mask: np.ndarray, count: np.ndarray = np.array(0)):
        if not mask[i, j]:
            mask[i, j] = True
            count += 1
            if padded_board[i, j] == 0:
                for di, dj in product([-1, 0, 1], [-1, 0, 1]):
                    self._flood_fill(i + di, j + dj, padded_board, mask, count)
        return count

    def _update(self) -> None:
        image = np.full_like(self._board, fill_value=10, dtype=int)
        image[self._visibility] = self._board[self._visibility]
        image[self._marks > 0] = self._marks[self._marks > 0] + 10
        self.rest_count = self.num_landmine - np.sum(self._marks == self.mark_flag)
        self.image = image
        if self.callback_fn is not None:
            self.callback_fn()

    def places_(self) -> np.ndarray:
        return np.argwhere(self._board == self.landmine)


class Board(BaseBoard):
    def __init__(self, num_landmine=10, shape: tuple = (9, 9)):
        super().__init__(num_landmine, shape=shape)
        self.h, self.w = shape
        assert self.h * self.w > num_landmine + 9, f"地雷太多了"

        self.step = 0
        self.last_click = None

    def reset(self, *args):
        self._board, places = generate_board(self.num_landmine, (self.h, self.w), Board.landmine, *args)
        self._visibility = np.zeros_like(self._board, dtype=bool)
        self._marks = np.zeros_like(self._board, dtype=int)
        self.finished = False
        self.state = self.STATE_GAMING
        self.step = 0
        self.last_click = None
        self._update()

    def click(self, i: int, j: int):
        if self.step == 0:
            self.reset(i, j)  # 第一次点击才初始化棋盘格，保证在点击位置范围3×3区域不生成地雷
        self.step += 1

        if not self._visibility[i, j] and not self.finished and self._marks[i, j] == 0 and 0 <= i < self.h and 0 <= j < self.w:
            if self._board[i, j] == 0:  # 点击到的为空格子，泛洪填充区域
                padded_board = pad(self._board)
                mask = pad(np.zeros_like(self._visibility), True)
                filled_cnt = self._flood_fill(i + 1, j + 1, padded_board, mask)
                self._visibility = np.bitwise_or(self._visibility, mask[1:-1, 1:-1])
            else:
                self._visibility[i, j] = True
                if self._board[i, j] == Board.landmine:
                    # 点击到地雷
                    self._lose()
                else:  # 点击到数字
                    self._check_win()
            self._update()
            self.last_click = (i, j)

    def render(self) -> np.ndarray:
        return self.image

    def mark(self, i: int, j: int):
        if not self._visibility[i, j] and not self.finished and 0 <= i < self.h and 0 <= j < self.w:
            self._marks[i, j] = (self._marks[i, j] + 1) % 3
            self._check_win()
            self._update()


if __name__ == "__main__":
    board = Board()
