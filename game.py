import os.path
from collections import namedtuple
from itertools import product

import pygame

from Board import Board
from optimizer import _solve
from solver import Solver


class Game:
    TITLE = "扫雷游戏"

    WHITE = (192, 192, 192)
    GRAY = (200, 200, 200)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    BLUE = (0, 0, 255)

    COLORS = {
        "1": (0, 0, 255),  # 蓝色
        "2": (0, 128, 0),  # 绿色
        "3": (255, 0, 0),  # 红色
        "4": (0, 0, 128),  # 深蓝色
        "5": (128, 0, 0),  # 栗色
        "6": (64, 224, 208),  # 青色
        "7": (0, 0, 0),  # 黑色
        "8": (128, 128, 128),  # 灰色
        "9": (255, 165, 0)  # 橙色
    }

    def __init__(self, board: Board, width=800, height=800, fontsize=1.0, top=50, auto_size=True):
        pygame.init()
        self.board = board

        if height / (self.board.h + 1) > top:
            top = int(height / self.board.w) + 1
        self.cell_size = min(width, height) // max(self.board.h, self.board.w)

        self.width = self.board.h * self.cell_size if auto_size else width
        self.height = self.board.w * self.cell_size + top if auto_size else height + top

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.TITLE)

        self.grid_size = (board.h, board.w)
        self.font_size = fontsize
        self.top_size = top

        self.resources = self.load_resources()

        self.running = False
        self.grid_x, self.grid_y = None, None
        self.button_pushed = False

        self.solver = Solver(self.board)

    def load_resources(self):
        root_path = 'resources'
        load = lambda filename: pygame.transform.scale(pygame.image.load(os.path.join(root_path, filename)),
                                                       (self.cell_size, self.cell_size))
        Resources = namedtuple('Resources',
                               ['block', 'landmine', 'flag', 'question_mark', 'block_rotated', 'font', 'font2', 'win', 'lose', 'tick',
                                'gaming'])
        block = load('block.png')
        block_rotated = pygame.transform.rotate(block, 180)
        landmine = load('landmine.png')
        flag = load('flag.png')
        question_mark = load('question_mark.png')
        win = load('win.png')
        lose = load('lose.png')
        gaming = load('gaming.png')
        tick = load('tick.png')
        font_size = int(self.font_size * 100 * self.cell_size / 88.88)
        font = pygame.font.SysFont(None, font_size)
        font2 = pygame.font.SysFont(None, font_size//4)
        return Resources(block, landmine, flag, question_mark, block_rotated, font, font2, win, lose, tick, gaming)

    def start(self):
        self.running = True
        cell_size = self.cell_size
        self.grid_x, self.grid_y = None, None
        button = pygame.Rect((self.width - cell_size) / 2, (self.top_size - cell_size) / 2, cell_size, cell_size)
        self.button_pushed = False

        self.update()
        pygame.display.flip()

        while self.running:
            flag_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    flag_changed = True
                    if event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        self.grid_x = mx // cell_size
                        self.grid_y = (my - self.top_size) // cell_size

                        if button.collidepoint((mx, my)):
                            self.button_pushed = True
                    elif event.button == 3:
                        mx, my = pygame.mouse.get_pos()
                        x = mx // cell_size
                        y = (my - self.top_size) // cell_size
                        if 0 <= x < self.board.h and 0 <= y < self.board.w:
                            board.mark(x, y)
                        if button.collidepoint((mx, my)):
                            self.solver.i_solve()
                            for place, p in self.solver.results.items():
                                if 0 <= p < 10e-6:
                                    if self.board.render()[place] == Board.mark_flag:
                                        self.board.mark(*place)
                                        self.board.mark(*place)
                                    self.board.click(*place)
                                elif p > 1 - 10e-6 and self.board.render()[place] == Board.mark_unknown:
                                    self.board.mark(*place)

                            # self.board.click(*place)

                elif event.type == pygame.MOUSEBUTTONUP:
                    flag_changed = True
                    if event.button == 1:
                        # 获取鼠标点击的位置
                        mx, my = pygame.mouse.get_pos()
                        # 计算点击的格子坐标
                        grid_x_release = mx // cell_size
                        grid_y_release = (my - self.top_size) // cell_size
                        if grid_y_release == self.grid_y and grid_x_release == self.grid_x:
                            if 0 <= self.grid_x < self.board.h and \
                                    0 <= self.grid_y < self.board.w:
                                board.click(self.grid_x, self.grid_y)
                        if button.collidepoint((mx, my)):
                            board.reset()
                            self.solver.results = None
                        self.grid_x, self.grid_y = None, None
                        self.button_pushed = False

            if flag_changed:
                self.update()
                pygame.display.flip()

        pygame.quit()

    def update(self, border=3):
        cell_size = self.cell_size
        font_size = self.font_size
        top_size = self.top_size
        screen = self.screen
        res = self.resources

        board = self.board.render()  # get board

        self.screen.fill(self.WHITE)

        rect = pygame.Rect((self.width - cell_size) / 2, (top_size - cell_size) / 2, cell_size, cell_size)
        if not self.button_pushed:  # draw state button
            screen.blit(res.block, rect.topleft)
        else:
            screen.blit(res.block_rotated, rect.topleft)

        if self.board.state == Board.STATE_GAMING:  # draw state emoji
            screen.blit(res.gaming, rect.topleft)
        elif self.board.state == Board.STATE_WIN:
            screen.blit(res.win, rect.topleft)
        elif self.board.state == Board.STATE_LOSE:
            screen.blit(res.lose, rect.topleft)

        for x, y in product(range(self.board.h), range(self.board.w)):
            rect = pygame.Rect(x * cell_size, y * cell_size + top_size, cell_size, cell_size)
            pygame.draw.rect(screen, self.GRAY, rect, border)

            if board[x][y] == Board.landmine:  # draw landmine
                if self.board.last_click == (x, y):
                    inner_rect = pygame.Rect(x * cell_size + border, y * cell_size + border + top_size, cell_size - border,
                                             cell_size - border)
                    pygame.draw.rect(screen, self.RED, inner_rect)
                screen.blit(res.landmine, rect.topleft)

            elif 10 > board[x][y] > 0:  # draw number
                color = self.COLORS.get(str(board[x][y]), self.BLACK)
                text = res.font.render(str(board[x][y]), True, color)
                screen.blit(text,
                            (x * cell_size + (cell_size - font_size) / 3,
                             y * cell_size + (cell_size - font_size) / 5 + top_size))

            elif board[x][y] != 0:  # draw unknown
                if (self.grid_x, self.grid_y) == (x, y) and board[x][y] == Board.mark_unknown:
                    screen.blit(res.block_rotated, rect.topleft)
                else:
                    screen.blit(res.block, rect.topleft)

                if board[x][y] == Board.mark_flag:
                    screen.blit(res.flag, rect.topleft)
                elif board[x][y] == Board.mark_question:
                    screen.blit(res.question_mark, rect.topleft)
                elif board[x][y] == Board.mark_edge:
                    pygame.draw.rect(screen, self.ORANGE, rect)

                if self.solver.results is not None:
                    if (x, y) in self.solver.results:
                        # if (x, y) == self.solver.best:
                        #     screen.blit(res.tick, rect.topleft)
                        prob = self.solver.results[(x, y)]
                        text = res.font2.render(f"{prob*100:.2f}", True, Game.BLUE)
                        screen.blit(text,
                                    (x * cell_size + (cell_size - font_size) / 3,
                                     y * cell_size + (cell_size - font_size) / 5 + top_size))


if __name__ == "__main__":
    board = Board(60, (19, 19))
    game = Game(board, width=1000, height=1000, top=30)
    game.start()
