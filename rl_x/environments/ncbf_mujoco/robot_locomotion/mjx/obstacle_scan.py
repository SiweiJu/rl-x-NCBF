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


# Obstacle shapes and motion types; the environment draws them with obstacles.shapes.weights and
# obstacles.motions.weights.
OBSTACLE_SHAPES = ("circle", "box", "circles", "l_shape")
OBSTACLE_MOTIONS = ("constant", "turning", "stop_and_go", "accelerating", "random_walk")


# Obstacles made of primitives. Each primitive has a kind, a centre, an angle and half extents
# (a circle uses half_extents[0] as its radius).
PRIMITIVE_NONE = 0
PRIMITIVE_CIRCLE = 1
PRIMITIVE_BOX = 2


def primitives_in_world(obstacle_pos, obstacle_heading, offsets, half_extents, angles, kinds):
    """Place each obstacle's primitives in the world frame and flatten them.

    Args:
        obstacle_pos: [K, 2] obstacle reference points in the world frame.
        obstacle_heading: [K] obstacle body orientations.
        offsets: [K, P, 2] primitive centres in the obstacle body frame.
        half_extents: [K, P, 2] primitive half extents (circle: radius, unused).
        angles: [K, P] primitive angles relative to the obstacle body.
        kinds: [K, P] PRIMITIVE_* values.

    Returns:
        centres [K*P, 2], angles [K*P], half_extents [K*P, 2], kinds [K*P].
    """
    cos_h, sin_h = jnp.cos(obstacle_heading)[:, None], jnp.sin(obstacle_heading)[:, None]
    world_offsets = jnp.stack([cos_h * offsets[..., 0] - sin_h * offsets[..., 1],
                               sin_h * offsets[..., 0] + cos_h * offsets[..., 1]], axis=-1)
    centres = obstacle_pos[:, None, :] + world_offsets
    world_angles = obstacle_heading[:, None] + angles
    return centres.reshape(-1, 2), world_angles.reshape(-1), half_extents.reshape(-1, 2), kinds.reshape(-1)


def _to_local(vectors_xy, cos_a, sin_a):
    """Rotate world-frame vectors [..., M, 2] into the frames of M primitives."""
    return (cos_a * vectors_xy[..., 0] + sin_a * vectors_xy[..., 1],
            -sin_a * vectors_xy[..., 0] + cos_a * vectors_xy[..., 1])


def primitive_range_scan(origin_xy, yaw, ray_angles, centres, angles, half_extents, kinds, max_range):
    """Planar range scan against circles and oriented boxes (flattened primitives, see primitives_in_world).

    Same conventions as obstacle_range_scan: max_range for no hit within range, 0 from inside a primitive.
    """
    ray_world = yaw + ray_angles
    directions = jnp.stack([jnp.cos(ray_world), jnp.sin(ray_world)], axis=-1)  # [N, 2]
    relative = origin_xy[None, :] - centres  # [M, 2]

    # Circles: |relative + t d|^2 = r^2
    radius = half_extents[:, 0]
    b = directions @ relative.T  # [N, M]
    c = jnp.sum(jnp.square(relative), axis=-1) - jnp.square(radius)
    discriminant = jnp.square(b) - c[None, :]
    sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t_near, t_far = -b - sqrt_discriminant, -b + sqrt_discriminant
    inside_circle = c[None, :] <= 0.0
    circle_hit = (discriminant >= 0.0) & (t_far >= 0.0)
    circle_distance = jnp.where(inside_circle, 0.0, jnp.where(circle_hit, jnp.maximum(t_near, 0.0), max_range))

    # Boxes: slab test in each box frame
    cos_a, sin_a = jnp.cos(angles), jnp.sin(angles)
    origin_x, origin_y = _to_local(relative, cos_a, sin_a)  # [M]
    dir_x, dir_y = _to_local(directions[:, None, :], cos_a[None, :], sin_a[None, :])  # [N, M]

    def slab(origin, direction, half):
        direction = jnp.where(jnp.abs(direction) < 1e-9, 1e-9, direction)
        t1, t2 = (-half - origin) / direction, (half - origin) / direction
        return jnp.minimum(t1, t2), jnp.maximum(t1, t2)

    enter_x, exit_x = slab(origin_x[None, :], dir_x, half_extents[None, :, 0])
    enter_y, exit_y = slab(origin_y[None, :], dir_y, half_extents[None, :, 1])
    t_enter, t_exit = jnp.maximum(enter_x, enter_y), jnp.minimum(exit_x, exit_y)
    box_hit = t_exit >= jnp.maximum(t_enter, 0.0)
    box_distance = jnp.where(box_hit, jnp.maximum(t_enter, 0.0), max_range)

    distance = jnp.where(kinds[None, :] == PRIMITIVE_CIRCLE, circle_distance,
                         jnp.where(kinds[None, :] == PRIMITIVE_BOX, box_distance, max_range))
    return jnp.clip(jnp.min(distance, axis=-1), 0.0, max_range)


def primitive_signed_distance(point_xy, centres, angles, half_extents, kinds):
    """Signed distance from a point to every primitive [M] (negative inside, inf for unused slots)."""
    relative = point_xy[None, :] - centres
    circle_distance = jnp.linalg.norm(relative, axis=-1) - half_extents[:, 0]
    local_x, local_y = _to_local(relative, jnp.cos(angles), jnp.sin(angles))
    q = jnp.stack([jnp.abs(local_x), jnp.abs(local_y)], axis=-1) - half_extents
    box_distance = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return jnp.where(kinds == PRIMITIVE_CIRCLE, circle_distance, jnp.where(kinds == PRIMITIVE_BOX, box_distance, jnp.inf))
