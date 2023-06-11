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
    render_rect = Rect(10, 10, 600, 600)
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


def render_image(screen: Surface, points, centers, c_points):
    render_rect = Rect(10, 10, 600, 600)

    render_points = c_points if c_points is not None else points
    pygame_image = surfarray.make_surface(render_points)
    scaled_image = transform.scale(pygame_image, render_rect.size)
    screen.blit(scaled_image, render_rect)
