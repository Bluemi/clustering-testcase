import pygame as pg


class Controller:
    def __init__(self):
        self.running = True
        self.show_original = True

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYDOWN:
            if event.key == 111:
                self.show_original = not self.show_original
            else:
                print(event)
