import random

from tqdm import tqdm

from templates import TargetFunction, Space
from templates import Optimizer


class Particle:
    def __init__(self, x_space: Space, speed_space: Space):
        self.x = x_space.sample()
        self.speed = speed_space.sample()
        self.best_x, self.best_y = None, -float('inf')

    def step(self, target_function: TargetFunction, copy_function):
        self.x += self.speed

        y = target_function(self.x)
        if y >= self.best_y:
            self.best_x, self.best_y = copy_function(self.x), y

    def change_direction(self, global_best_x, omiga, c1, c2):
        self.speed = self.speed * omiga + random.random() * c1 * (self.best_x - self.x) + random.random() * c2 * (global_best_x - self.x)

    def __lt__(self, other: 'Particle'):
        return self.best_y < other.best_y


class ParticleSwarmOptimization(Optimizer):
    def __init__(self, num_particles, max_iteration, speed_space: Space, copy_function, init_omiga=0.9, end_omiga=0.4, c1=2.0, c2=2.0):
        self.num_particles = num_particles
        self.max_iterations = max_iteration
        self.speed_space = speed_space
        self.omiga = init_omiga
        self.delta_omiga = (init_omiga - end_omiga) / max_iteration
        self.copy_function = copy_function if copy_function is not None else lambda x: x
        self.c1 = c1
        self.c2 = c2

    def optimize(self, target_function: TargetFunction, x_space: Space, show_progress=False):
        particles = [Particle(x_space, self.speed_space) for _ in range(self.num_particles)]
        best_particle = None
        pbar = tqdm if show_progress else lambda x: x
        for _ in pbar(range(self.max_iterations)):
            best_particle = max(particles)
            for particle in particles:
                particle.step(target_function, self.copy_function)
                particle.change_direction(best_particle.best_x, self.omiga, self.c1, self.c2)
            self.omiga -= self.delta_omiga
        return best_particle.best_x, best_particle.best_y

