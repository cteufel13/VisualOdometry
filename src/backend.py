"""Map maintenance logic."""

from datatypes import Frame, State


def create_new_map_points(frame_curr: Frame, frame_ref: Frame, state: State) -> None:
    """
    Triangulate candidates that have sufficient parallax.

    Args:
        frame_curr: The current frame (with 2D feature observations).
        frame_ref: The reference frame (where candidates were first seen).
        state: The global state object to update.

    """
    # TODO: Map Expansion Logic
    pass


def cull_outliers(state: State) -> None:
    """
    Clean up the map.

    Args:
        state: Global state object.

    """
    # TODO: Map Maintenance
    pass
