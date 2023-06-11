#!/usr/bin/env python3


import pygame as pg
import numpy as np

import distributions
from algorithms.linear_quantisation import LinearQuantization
from controller import Controller
from render import render

DEFAULT_SCREEN_SIZE = np.array([1280, 720])


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    controller = Controller()

    num_points = 1000
    # distribution = distributions.UniformDistribution(num_points, 2)
    distribution = distributions.CenteredDistribution(5, num_points, 2, seed=0)
    points = distribution.generate_points()

    algorithm = LinearQuantization(5**2)
    centers, clustered_points = algorithm.cluster(points)
    # centers = None
    # clustered_points = None

    while controller.running:
        events = [pg.event.wait()]
        for event in events + pg.event.get():
            controller.handle_event(event)

        if controller.show_original:
            render(screen, points)
        else:
            render(screen, points, centers, clustered_points)
        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()

