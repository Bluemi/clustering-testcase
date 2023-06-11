import os

import numpy as np
from PIL import Image

ALLOWED_FILE_TYPES = {'.png', '.jpg'}


class ImageLoader:
    def __init__(self):
        self.image_index = 0
        self.image_paths = [p for p in os.listdir('images') if os.path.splitext(p)[1] in ALLOWED_FILE_TYPES]
        self.image = self.load_image()

    def load_image(self):
        path = os.path.join('images', self.image_paths[self.image_index])
        # noinspection PyTypeChecker
        return np.array(Image.open(path), dtype=np.int32)

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.image = self.load_image()
        return self.image
