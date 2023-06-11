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
    distribution = distributions.GaussDistribution(num_points, 2)
    points = distribution.generate_points()

    algorithm = LinearQuantization(64)
    centers, clustered_points = algorithm.cluster(points)

    while controller.running:
        events = [pg.event.wait()]
        for event in events + pg.event.get():
            controller.handle_event(event)

        render(screen, points, centers, clustered_points)
        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()

