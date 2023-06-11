#!/usr/bin/env python3


import pygame as pg
import numpy as np

from controller import Controller
from render import render

DEFAULT_SCREEN_SIZE = np.array([1280, 720])


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    controller = Controller()

    while controller.running:
        events = [pg.event.wait()]
        for event in events + pg.event.get():
            controller.handle_event(event)

        render(screen)
        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()

