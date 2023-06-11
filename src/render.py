from typing import Optional

import numpy as np
from pygame import Surface, Rect, draw, Color, transform, surfarray

from utils import BLACK, gray


def render(screen: Surface, points: np.ndarray, centers: Optional[np.ndarray] = None,
           clustered_points: Optional[np.ndarray] = None):
    screen.fill(BLACK)
    if points.shape[-1] == 2:
        render_2d_points(screen, points, centers, clustered_points)
    elif points.shape[-1] == 3:
        render_image(screen, points, centers, clustered_points)


def render_2d_points(screen: Surface, points: np.ndarray, centers, c_points):
    render_rect = Rect(10, 10, 800, 800)
    draw.rect(screen, gray(40), render_rect)

    # to render coordinates
    render_points = (points * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)

    if centers is not None and c_points is not None:
        render_c_points = (c_points * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)
        render_centers = (centers * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)

        for r_point, r_clustered, clustered in zip(render_points, render_c_points, c_points):
            color = Color(127, int(clustered[0]), int(clustered[1]))
            draw.circle(screen, color + gray(30), r_point, 2)
            draw.line(screen, color - gray(30), r_point, r_clustered)

        for r_center, center in zip(render_centers, centers):
            # color = Color(127, int(center[0]), int(center[1]))
            color = gray(220)
            draw.circle(screen, color, r_center, 3)
    else:
        for r_point in render_points:
            color = gray(220)
            draw.circle(screen, color, r_point, 2)


def render_image(screen: Surface, pixels, centers, clustered_pixels):
    render_rect = Rect(10, 10, 800, 800)

    render_pixels = clustered_pixels if clustered_pixels is not None else pixels
    pygame_image = surfarray.make_surface(render_pixels)
    image_width = render_rect.width
    image_height = render_rect.height
    if pygame_image.get_width() > pygame_image.get_height():
        image_height = render_rect.height * (pygame_image.get_height() / pygame_image.get_width())
    elif pygame_image.get_height() > pygame_image.get_width():
        image_width = render_rect.width * (pygame_image.get_width() / pygame_image.get_height())
    scaled_image = transform.scale(pygame_image, (image_width, image_height))
    screen.blit(scaled_image, render_rect)

    if centers is not None:
        palette = create_image_with_colors(centers)
        pygame_palette = surfarray.make_surface(palette)
        screen.blit(pygame_palette, Rect(10, 820, palette.shape[1], palette.shape[0]))


def create_image_with_colors(colors: np.ndarray):
    height = 40
    width = 800 // colors.shape[0]
    color_images = []
    for c in colors:
        color_images.append(np.full((width, height, 3), c))
    return np.concatenate(color_images, axis=0)
