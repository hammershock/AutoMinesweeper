from abc import abstractmethod


class TargetFunction:
    @abstractmethod
    def __call__(self, input_x) -> float:
        pass


class Space:
    @abstractmethod
    def sample(self):
        pass


class Optimizer:
    @abstractmethod
    def optimize(self, target_function: TargetFunction, x_space: Space):
        pass


class Disturbance:
    @abstractmethod
    def __call__(self, input_x, temperature: float = None):
        pass
