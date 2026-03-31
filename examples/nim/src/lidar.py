import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RayCastHit:
    """
    Result of a ray-cast.

    Attributes:
        hit_point: tuple[float, float] - coordinates of the hit or end point
        distance: float - distance from origin to hit_point
        hit: bool - True if an occupied cell was hit

    """

    hit_point: tuple[int, int]
    distance: float
    hit: bool


def image_to_occupancy(
    image_array: NDArray[np.float64],
    threshold: float = 0.5,
) -> NDArray[np.uint8]:
    """
    Convert grayscale image (0..1) to boolean occupancy map.

    True means occupied (wall), False means free.

    Args:
        image_array: 2D array of shape (H,W) with values in [0,1]
        threshold: value below which is considered occupied

    Returns:
        2D boolean array of shape (H,W) where True=occupied, False=free.

    """
    return (image_array < threshold).astype(np.uint8)


def ray_cast(
    occupancy: NDArray[np.uint8],
    origin: tuple[int, int],
    angle: float,
    max_range: float,
) -> RayCastHit:
    """
    Cast a single ray using Bresenham on the occupancy grid.

    Args:
        occupancy: 2D array of shape (H,W) with 1 for occupied and 0 for free
        origin: (x,y) coordinates of ray origin in grid space (can be float)
        angle: direction of ray in radians (0 = right, pi/2 = down)
        max_range: maximum range to cast the ray

    Returns:
        RayCastHit dataclass

    """
    h, w = occupancy.shape
    ox, oy = origin

    # Endpoint of the ray in continuous coordinates
    x_end = ox + max_range * math.cos(angle)
    y_end = oy + max_range * math.sin(angle)

    # Convert origin and end to integer grid cells for Bresenham
    ix0 = round(ox)
    iy0 = round(oy)
    ix1 = round(x_end)
    iy1 = round(y_end)

    # Iterate cells along the line
    for cx, cy in _bresenham_line(ix0, iy0, ix1, iy1):
        # Check bounds
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            # Outside image: clamp to border and return no-hit
            clamped_x = min(max(cx, 0), w - 1)
            clamped_y = min(max(cy, 0), h - 1)
            # compute distance to clamped cell center
            dist = math.hypot(clamped_x - ox, clamped_y - oy)
            return RayCastHit(
                hit_point=(clamped_x, clamped_y), distance=dist, hit=False
            )

        if occupancy[cy, cx]:
            # Hit an occupied cell; report the cell center as hit point
            hit_x = float(cx)
            hit_y = float(cy)
            dist = math.hypot(hit_x - ox, hit_y - oy)
            return RayCastHit(hit_point=(cx, cy), distance=dist, hit=True)

    # No hit within range
    return RayCastHit(hit_point=(ix1, iy1), distance=max_range, hit=False)


def simulate_lidar_scan(
    occupancy: NDArray[np.uint8],
    origin: tuple[int, int],
    *,
    heading: float | None = None,
    num_beams: int = 36,
    fov: float = 2 * math.pi,
    max_range: float = 50.0,
) -> list[RayCastHit]:
    """
    Simulate a 2D lidar scan from origin.

    Args:
        occupancy: 2D array of shape (H,W) with 1 for occupied and 0 for free
        origin: (x,y) coordinates of scan origin in grid space (can be float)
        heading: optional angle in radians to center the scan around (if fov < 2pi)
        num_beams: how many beams in the scan
        fov: field of view for the scan (radians, default 2pi for full circle)
        max_range: maximum range for the lidar beams

    Returns:
        list of RayCastHit for each beam.

    Raises:
        ValueError: if fov > 2*pi

    """
    if fov > 2 * math.pi:
        raise ValueError("fov cannot be greater than 2*pi")

    start_angle = -fov / 2
    start_angle += heading if heading is not None else 0.0
    return [
        ray_cast(
            occupancy,
            origin,
            start_angle + (i / max(1, num_beams - 1)) * fov,
            max_range,
        )
        for i in range(num_beams)
    ]


def sample_along_path(
    occupancy: NDArray[np.uint8],
    path: Iterable[tuple[int, int, float]],
    *,
    samples_per_pose: int = 5,
    num_beams: int = 36,
    fov: float = math.pi * 2,
    max_range: float = 50.0,
) -> list[tuple[tuple[int, int], bool]]:
    """
    Generate dataset samples by moving along a path and doing lidar scans.

    For each path pose, sample `samples_per_pose` beams chosen randomly among the
    `num_beams` beams, produce (x,y) point coordinates in grid space and a label
    (1 if beam hit wall within max_range else 0).

    Args:
        occupancy: 2D array of shape (H,W) with 1 for occupied and 0 for free
        path: iterable of (x,y,theta) poses to simulate the scan from
        samples_per_pose: how many beams to sample from each pose's scan
        num_beams: how many beams in the full scan (before sampling)
        fov: field of view for the scan (radians, default 2pi for full circle)
        max_range: maximum range for the lidar beams

    Returns:
        List of tuples: [((x,y), hit), ...] where (x,y) are the coordinates of the beam hit (or max range endpoint)
            and hit is a boolean label for hitting a wall.

    """
    samples: list[tuple[tuple[int, int], bool]] = []
    for pose in path:
        # pose can be (x,y) or (x,y,theta)
        ox, oy, heading = pose
        scan_hits = simulate_lidar_scan(
            occupancy=occupancy,
            origin=(ox, oy),
            heading=heading,
            num_beams=num_beams,
            fov=fov,
            max_range=max_range,
        )

        # Randomly sample beams
        beam_indices = list(range(len(scan_hits)))
        chosen = (
            random.sample(beam_indices, samples_per_pose)
            if samples_per_pose < len(beam_indices)
            else beam_indices
        )

        for bi in chosen:
            hit = scan_hits[bi]
            samples.append((hit.hit_point, hit.hit))

    return samples


def densify_path(
    path: Iterable[tuple[int, int]],
    spacing: float,
) -> list[tuple[int, int]]:
    """
    Densify a path by interpolating points along segments.

    Returns a new path where points are interpolated along segments so that consecutive points are approximately
    `spacing` pixels apart. If spacing <= 0, returns the original path as a list.

    Args:
        path: Iterable of (x,y) points representing the path to densify
        spacing: Desired spacing between consecutive points in the output path. If <= 0, no densification is done.

    Returns:
        List of (x,y) points along the path, including original endpoints, with added points in between as needed.

    """
    if spacing is None or spacing <= 0:
        return list(path)

    pts = list(path)

    if len(pts) == 0:
        return []

    if len(pts) == 1:
        return pts

    out: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg_len == 0:
            continue
        num = max(1, math.ceil(seg_len / spacing))
        for k in range(num):
            t = k / num
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            out.append((int(x), int(y)))
    # Remove duplicates that may arise from bad spacing
    out = list(dict.fromkeys(out))
    # include last point
    out.append(pts[-1])
    return out


def compute_headings(
    path: Iterable[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    """
    Given a sequence of (x,y) points, return list of (x,y,theta) where theta is the heading.

    The heading assumed to be movement from previous point to current point. If the path has a single point, theta=0.

    """
    pts = [tuple(p) for p in path]
    out: list[tuple[int, int, float]] = []
    for i, p in enumerate(pts):
        if i > 0:
            nx = p[0] - pts[i - 1][0]
            ny = p[1] - pts[i - 1][1]
            theta = math.atan2(ny, nx)
        elif i < len(pts) - 1:
            nx = pts[i + 1][0] - p[0]
            ny = pts[i + 1][1] - p[1]
            theta = math.atan2(ny, nx)
        else:
            theta = 0.0
        out.append((p[0], p[1], theta))
    return out


# Path generators
def random_walk_path(
    start: tuple[int, int],
    num_steps: int,
    step_size: int,
    bounds: tuple[int, int],
) -> list[tuple[int, int]]:
    """Generate a random walk starting at `start`. `bounds` is (width, height) to clamp the path."""
    w, h = bounds
    x, y = start
    path = [(x, y)]
    for _ in range(num_steps - 1):
        angle = random.random() * 2 * math.pi
        x = int(min(max(x + math.cos(angle) * step_size, 0.0), w - 1.0))
        y = int(min(max(y + math.sin(angle) * step_size, 0.0), h - 1.0))
        path.append((x, y))
    return path


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> Iterable[tuple[int, int]]:
    """
    Yield integer grid cells along a line from (x0,y0) to (x1,y1) (inclusive).

    Classic Bresenham integer algorithm.

    Args:
        x0: start cell x-coordinate
        y0: start cell y-coordinate
        x1: end cell x-coordinate
        y1: end cell y-coordinate

    Yields:
        (cx, cy) - integer coordinates of each cell along the line

    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
