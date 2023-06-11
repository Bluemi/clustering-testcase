import numpy as np
from pygame import Surface, Rect, draw, Color

from utils import BLACK, gray


def render(screen: Surface, points: np.ndarray, centers: np.ndarray, clustered_points: np.ndarray):
    screen.fill(BLACK)
    if points.shape[-1] == 2:
        render_2d_points(screen, points, centers, clustered_points)
    elif points.shape[-1] == 3:
        render_image(points)


def render_2d_points(screen: Surface, points: np.ndarray, centers, c_points):
    render_rect = Rect(10, 10, 600, 600)
    draw.rect(screen, gray(40), render_rect)

    # to render coordinates
    render_points = (points * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)
    render_c_points = (c_points * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)
    render_centers = (centers * ((render_rect.width - 20) / 255) + (render_rect.left + 10)).astype(int)

    for point, r_point, clustered in zip(points, render_points, render_c_points):
        color = Color(127, int(point[0]), int(point[1]))
        draw.circle(screen, color, r_point, 2)
        draw.line(screen, color, r_point, clustered)

    for r_center, center in zip(render_centers, centers):
        color = Color(127, int(center[0]), int(center[1]))
        draw.circle(screen, color, r_center, 3)


def render_image(points):
    pass
