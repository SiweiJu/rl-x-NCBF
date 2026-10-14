import jax.numpy as jnp


def obstacle_range_scan(origin_xy, yaw, ray_angles, obstacle_pos, obstacle_radius, max_range):
    """Planar range scan against circular obstacles.

    Rays start at origin_xy and point along yaw + ray_angles. Each ray returns the distance to the
    nearest obstacle boundary it hits, or max_range when it hits nothing within range. An origin
    inside an obstacle gives 0. Nearer obstacles occlude farther ones.

    Args:
        origin_xy: [2] ray origin in the world frame.
        yaw: scalar heading of the scanner in the world frame.
        ray_angles: [N] ray angles relative to the heading.
        obstacle_pos: [K, 2] obstacle centres in the world frame.
        obstacle_radius: [K] obstacle radii.
        max_range: maximum range of the scanner.

    Returns:
        [N] ranges in [0, max_range].
    """
    angles = yaw + ray_angles
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # [N, 2]
    offset = origin_xy[None, :] - obstacle_pos  # [K, 2], origin relative to each centre
    # |offset + t * d|^2 = r^2  ->  t^2 + 2 b t + c = 0 with b = offset . d, c = |offset|^2 - r^2
    b = directions @ offset.T  # [N, K]
    c = jnp.sum(jnp.square(offset), axis=-1) - jnp.square(obstacle_radius)  # [K]
    discriminant = jnp.square(b) - c[None, :]
    sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t_near = -b - sqrt_discriminant
    t_far = -b + sqrt_discriminant
    inside = c[None, :] <= 0.0
    hit = (discriminant >= 0.0) & (t_far >= 0.0)
    distance = jnp.where(inside, 0.0, jnp.where(t_near >= 0.0, t_near, 0.0))
    distance = jnp.where(hit | inside, distance, max_range)
    return jnp.clip(jnp.min(distance, axis=-1), 0.0, max_range)
