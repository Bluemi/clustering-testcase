#!/usr/bin/env python3


import pygame as pg
import numpy as np


DEFAULT_SCREEN_SIZE = np.array([1280, 720])


def handle_event(event):
    running = True
    if event.type == pg.QUIT:
        running = False
    return running


def main():
    pg.init()
    screen = pg.display.set_mode(DEFAULT_SCREEN_SIZE)
    running = True

    while running:
        events = [pg.event.wait()]
        for event in events + pg.event.get():
            running = handle_event(event)

        screen.fill(pg.Color(0, 0, 0))
        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()

