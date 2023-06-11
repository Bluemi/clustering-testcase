import pygame as pg

from distributions import DISTRIBUTIONS


class Controller:
    def __init__(self):
        self.running = True
        self.show_original = True
        self.new_points = False

        # distribution
        self.current_distribution_index = 0
        self.distribution = None
        self.build_distribution()

    def build_distribution(self):
        self.distribution = DISTRIBUTIONS[self.current_distribution_index](num_points=100, num_dims=2, seed=0)

    def next_distribution(self):
        self.current_distribution_index = (self.current_distribution_index + 1) % len(DISTRIBUTIONS)
        self.build_distribution()

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
