#!/usr/bin/env python3
import time

import pygame as pg
import numpy as np

from ImageLoader import ImageLoader
from algorithms import ALGORITHMS
from render import render
from distributions import DISTRIBUTIONS


DEFAULT_SCREEN_SIZE = np.array([1400, 880])
NUM_CHUNKS = [1, 2, 4, 8, 9, 16, 27, 32, 64, 128, 216, 256, 512, 1024]


class Model:
    def __init__(self, screen: pg.Surface):
        self.screen = screen
        self.running = True
        self.show_cluster = False

        self.points = None
        self.centers = None
        self.clustered_points = None

        # algorithm
        self.num_chunks_index = 5
        self.algorithm_index = 0
        self.algorithm = self.build_algorithm()
        self.error = None
        self.calculation_time = 0

        self.show_2d = True
        # distribution 2d
        self.current_distribution_2d_index = 0
        self.distribution_2d = None
        self.build_distribution_2d()

        # images
        self.image_loader = ImageLoader()

    def build_algorithm(self):
        return ALGORITHMS[self.algorithm_index](NUM_CHUNKS[self.num_chunks_index])

    def next_algorithm(self):
        self.algorithm_index = (self.algorithm_index + 1) % len(ALGORITHMS)
        self.algorithm = self.build_algorithm()

    def prev_algorithm(self):
        self.algorithm_index = self.algorithm_index - 1
        if self.algorithm_index < 0:
            self.algorithm_index = len(ALGORITHMS) - 1
        self.algorithm = self.build_algorithm()

    def build_distribution_2d(self):
        self.distribution_2d = DISTRIBUTIONS[self.current_distribution_2d_index](num_points=400, num_dims=2, seed=0)

    def next_distribution(self):
        if self.show_2d:
            self.current_distribution_2d_index = (self.current_distribution_2d_index + 1) % len(DISTRIBUTIONS)
            self.build_distribution_2d()
        else:
            self.points = self.image_loader.next_image()

    def prev_distribution(self):
        if self.show_2d:
            self.current_distribution_2d_index -= 1
            if self.current_distribution_2d_index < 0:
                self.current_distribution_2d_index = len(DISTRIBUTIONS) - 1
            self.build_distribution_2d()
        else:
            self.points = self.image_loader.prev_image()

    def generate_and_cluster(self):
        if self.show_2d:
            self.points = self.distribution_2d.generate_points()
        else:
            self.points = self.image_loader.image

        self.cluster()

    def cluster(self):
        if len(self.points.shape) == 2:
            t1 = time.time()
            self.centers, self.clustered_points = self.algorithm.cluster(self.points)
            self.error = np.mean(np.linalg.norm(self.points - self.clustered_points, axis=1))
            t2 = time.time()
            self.calculation_time = t2 - t1
        else:
            old_shape = self.points.shape
            assert old_shape[-1] == 3
            points = self.points.reshape(-1, 3)
            t1 = time.time()
            self.centers, clustered_points = self.algorithm.cluster(points)
            self.error = np.mean(np.linalg.norm(points - clustered_points, axis=1))
            t2 = time.time()
            self.calculation_time = t2 - t1
            self.clustered_points = clustered_points.reshape(old_shape)

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYDOWN:
            if event.unicode == 'c':
                self.show_cluster = not self.show_cluster
            elif event.unicode == 'r':
                self.generate_and_cluster()
            elif event.unicode == 'n':
                self.next_distribution()
                self.generate_and_cluster()
            elif event.unicode == 'N':
                self.prev_distribution()
                self.generate_and_cluster()
            elif event.unicode == 'a':
                self.next_algorithm()
                self.cluster()
            elif event.unicode == 'A':
                self.prev_algorithm()
                self.cluster()
            elif event.unicode == '2':
                self.show_2d = True
                self.generate_and_cluster()
            elif event.unicode == '3':
                self.show_2d = False
                self.generate_and_cluster()
            elif event.unicode == '+':
                self.num_chunks_index = min(len(NUM_CHUNKS)-1, self.num_chunks_index+1)
                self.algorithm = self.build_algorithm()
                self.cluster()
            elif event.unicode == '-':
                self.num_chunks_index = max(0, self.num_chunks_index - 1)
                self.algorithm = self.build_algorithm()
                self.cluster()
            elif event.key == 27:
                self.running = False
            else:
                # print(event)
                pass

    def run(self):
        self.generate_and_cluster()

        while self.running:
            events = [pg.event.wait()]
            for event in events + pg.event.get():
                self.handle_event(event)

            if self.show_2d:
                source_name = self.distribution_2d.name()
            else:
                source_name = self.image_loader.image_paths[self.image_loader.image_index]
            if self.show_cluster:
                render(self.screen, self.points, self.algorithm.name(), source_name, NUM_CHUNKS[self.num_chunks_index],
                       self.error, self.calculation_time, self.centers, self.clustered_points)
            else:
                render(self.screen, self.points, self.algorithm.name(), source_name,
                       NUM_CHUNKS[self.num_chunks_index], self.error, self.calculation_time)
            pg.display.flip()

        pg.quit()


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    main_instance = Model(screen)
    main_instance.run()


if __name__ == '__main__':
    main()

