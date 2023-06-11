#!/usr/bin/env python3


import pygame as pg
import numpy as np

from ImageLoader import ImageLoader
from algorithms.linear_quantisation import LinearQuantization
from render import render
from distributions import DISTRIBUTIONS


DEFAULT_SCREEN_SIZE = np.array([1280, 720])


class Model:
    def __init__(self, screen: pg.Surface):
        self.screen = screen
        self.running = True
        self.show_cluster = False

        self.algorithm = LinearQuantization(5**2)
        self.points = None
        self.centers = None
        self.clustered_points = None

        self.show_2d = True
        # distribution 2d
        self.current_distribution_2d_index = 0
        self.distribution_2d = None
        self.build_distribution_2d()

        # images
        self.image_loader = ImageLoader()

    def build_distribution_2d(self):
        self.distribution_2d = DISTRIBUTIONS[self.current_distribution_2d_index](num_points=100, num_dims=2, seed=0)

    def next_distribution(self):
        self.current_distribution_2d_index = (self.current_distribution_2d_index + 1) % len(DISTRIBUTIONS)
        self.build_distribution_2d()

    def generate_and_cluster(self):
        self.points = self.distribution_2d.generate_points()
        self.centers, self.clustered_points = self.algorithm.cluster(self.points)

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYDOWN:
            if event.key == 99:
                self.show_cluster = not self.show_cluster
            elif event.key == 110:
                self.generate_and_cluster()
            elif event.key == 100:
                self.next_distribution()
                self.generate_and_cluster()
            elif event.key == 50:
                self.show_2d = True
            elif event.key == 51:
                self.show_2d = False
            elif event.key == 27:
                self.running = False
            else:
                print(event)

    def run(self):
        self.generate_and_cluster()

        while self.running:
            events = [pg.event.wait()]
            for event in events + pg.event.get():
                self.handle_event(event)

            if self.show_cluster:
                render(self.screen, self.points, self.centers, self.clustered_points)
            else:
                render(self.screen, self.points)
            pg.display.flip()

        pg.quit()


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    main_instance = Model(screen)
    main_instance.run()


if __name__ == '__main__':
    main()

