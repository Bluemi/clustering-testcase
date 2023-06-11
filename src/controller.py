import pygame as pg


class Controller:
    def __init__(self):
        self.running = True

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False