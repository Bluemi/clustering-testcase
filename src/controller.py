import pygame as pg


class Controller:
    def __init__(self):
        self.running = True
        self.show_original = True
        self.new_points = False

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.KEYDOWN:
            if event.key == 111:
                self.show_original = not self.show_original
            elif event.key == 110:
                self.new_points = True
            elif event.key == 27:
                self.running = False
            else:
                print(event)
