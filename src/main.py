#!/usr/bin/env python3


import pygame as pg
import numpy as np

from algorithms.linear_quantisation import LinearQuantization
from render import render
from distributions import DISTRIBUTIONS


DEFAULT_SCREEN_SIZE = np.array([1280, 720])


class Model:
    def __init__(self, screen: pg.Surface):
        self.screen = screen
        self.running = True
        self.show_original = True
        self.new_points = False

        self.show_2d = True
        # distribution 2d
        self.current_distribution_2d_index = 0
        self.distribution_2d = None
        self.build_distribution_2d()

    def build_distribution_2d(self):
        self.distribution_2d = DISTRIBUTIONS[self.current_distribution_2d_index](num_points=100, num_dims=2, seed=0)

    def next_distribution(self):
        self.current_distribution_2d_index = (self.current_distribution_2d_index + 1) % len(DISTRIBUTIONS)
        self.build_distribution_2d()

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYDOWN:
            if event.key == 111:
                self.show_original = not self.show_original
            elif event.key == 110:
                self.new_points = True
            elif event.key == 100:
                self.next_distribution()
                self.new_points = True
            elif event.key == 27:
                self.running = False
            else:
                print(event)

    def run(self):
        points = self.distribution_2d.generate_points()

        algorithm = LinearQuantization(5**2)
        centers, clustered_points = algorithm.cluster(points)

        while self.running:
            events = [pg.event.wait()]
            for event in events + pg.event.get():
                self.handle_event(event)

            if self.new_points:
                points = self.distribution_2d.generate_points()
                centers, clustered_points = algorithm.cluster(points)
                self.new_points = False

            if self.show_original:
                render(self.screen, points)
            else:
                render(self.screen, points, centers, clustered_points)
            pg.display.flip()

        pg.quit()


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    main_instance = Model(screen)
    main_instance.run()


if __name__ == '__main__':
    main()

