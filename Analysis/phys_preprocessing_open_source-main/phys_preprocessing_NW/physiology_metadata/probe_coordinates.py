"""Functions for computing probe coordinates.

Grid coordinates: Origin is reference  point of surface of brain under middle
grid hole (middle in both left-right and anterior-posterior). The planes of both
neuropixel and V-probe recording areas pass through this point.

Generally L-R axis is x (index 0), P-A axis is y (index 1), V-D axis is z (index
2).

Stereotactic coordinate are w.r.t. ear bar zero and level so that the bottom of
the eye orbit is at the same z as ear bar zero. Ray between posterior and
anterior commissure is parallel with this line.

All grid holes are 1-indexed.
"""

import numpy as np


class V_PROBE_PARAMS():

    ################################################################################
    ####  V-PROBE CHANNEL MEASUREMENTS, 64 CHANNELS, 50um SPACING
    ################################################################################

    # Distance from probe tip to first channel
    V_PROBE_TIP_TO_FIRST_CHANNEL = 0.15
    # Distance from probe tip to last (base) channel
    V_PROBE_TIP_TO_LAST_CHANNEL = 63 * 0.05 + V_PROBE_TIP_TO_FIRST_CHANNEL
    # Distance from probe tip to midpoint of channels
    V_PROBE_TIP_TO_MIDPOINT = 0.5 * (
        V_PROBE_TIP_TO_FIRST_CHANNEL + V_PROBE_TIP_TO_LAST_CHANNEL)


def _grid_from_other_side(phys_params, x):
    """Measure grid hole one-indexed from opposite side.
    
    Assumes square grid, which both monkeys have.
    """
    if x is None:
        raise ValueError('Invalid x is None.')
    return phys_params.NUM_GRID_HOLES + 1 - x


def _grid_holes_to_grid_coords_axial(phys_params,
                                     from_posterior=None,
                                     from_anterior=None,
                                     from_left=None,
                                     from_right=None):
    """Convert grid holes to grid coordinates in axial plane.

    Grid coordinates are with respect to the middle hole in the posterior row of
    the grid. Note that from_posterior and from_left are 1-indexed, so must
    switch to zero-indexing.
    """
    if from_posterior is None:
        from_posterior = _grid_from_other_side(phys_params, from_anterior)
    if from_left is None:
        from_left = _grid_from_other_side(phys_params, from_right)

    grid_coords_axial = phys_params.GRID_HOLE_SPACING * np.array([
        (from_left - 0.5 * (phys_params.NUM_GRID_HOLES + 1)),
        (from_posterior - 0.5 * (phys_params.NUM_GRID_HOLES + 1)),
    ])

    return grid_coords_axial


def _grid_coords_to_stereotaxic_coords(phys_params, grid_coords):
    """Convert grid coordinates to stereotaxic coordinates."""
    
    # Get grid angles in radians
    theta_coronal = np.radians(phys_params.GRID_ANGLE_CORONAL)
    theta_saggital = np.radians(phys_params.GRID_ANGLE_SAGGITAL)
    theta_axial = np.radians(phys_params.GRID_ANGLE_AXIAL)

    # Rotation from grid to stereo in coronal plane
    rotation_mat_coronal = np.array([
        [np.cos(theta_coronal), 0., -1. * np.sin(theta_coronal)],
        [0., 1., 0.],
        [np.sin(theta_coronal), 0., np.cos(theta_coronal)],
    ])

    # Rotation from grid to stereo in saggital plane
    rotation_mat_saggital = np.array([
        [1., 0., 0.],
        [0., np.cos(theta_saggital), -1. * np.sin(theta_saggital)],
        [0., np.sin(theta_saggital), np.cos(theta_saggital)],
    ])

    # Rotation from grid to stereo in axial plane
    rotation_mat_axial = np.array([
        [np.cos(theta_axial), -1. * np.sin(theta_axial), 0.],
        [np.sin(theta_axial), np.cos(theta_axial), 0.],
        [0., 0., 1.],
    ])

    # Total rotation from grid to stereo. Rotations are small enough that
    # effects of  non-commutativity of matrix multiplication are negligible.
    rotation_grid_to_stereo = np.matmul(
        rotation_mat_axial,
        np.matmul(rotation_mat_saggital, rotation_mat_coronal),
    )

    # Rotate grid_coords to stereotaxic coordinates
    rotated_grid_coords = np.dot(rotation_grid_to_stereo, grid_coords)

    # Translate from reference point to origin in stereotaxis coordinates
    translation = np.array([
        phys_params.REFERENCE_SURFACE_POINT_LR,
        phys_params.REFERENCE_SURFACE_POINT_PA,
        phys_params.REFERENCE_SURFACE_POINT_VD,
    ])
    stereo_coords = rotated_grid_coords + translation

    return stereo_coords


def _line_plane_intersept(plane_normal, plane_point, line_vector, line_point):
    """Compute intersept between line and plane in 3D space."""
    # Normalize plane normal and line vector
    plane_normal /= np.linalg.norm(plane_normal)
    line_vector /= np.linalg.norm(line_vector)

    # Compute intersept
    parallelism = np.dot(plane_normal, line_vector)
    point_difference = line_point - plane_point
    intersept = (
        line_point - 
        (line_vector * np.dot(plane_normal, point_difference) / parallelism)
    )

    return intersept


def _get_brain_surface_normal(phys_params, probe):
    """Get vector normal to brain surface pointing dorsally.

    In stereotaxis coordinates.
    
    Normal is slightly different for area around V-probe entry and area around
    neuropixel entry, so condition on the probe.

    Args:
        probe: String. 'v_probe' or 'np'. Which recording area to compute the
            surface normal with respect to.

    Returns:
        surface_normal: Float array of shape (3,). Unit vector normal to brain
            surface pointing dorsally.
    """
    # Get surface angles in saggital and coronal planes
    if probe == 'np':
        surface_angle_saggital = phys_params.SURFACE_ANGLE_SAGGITAL_NEUROPIXEL
        surface_angle_coronal = phys_params.SURFACE_ANGLE_CORONAL_NEUROPIXEL
    elif probe == 'v_probe':
        surface_angle_saggital = phys_params.SURFACE_ANGLE_SAGGITAL_V_PROBE
        surface_angle_coronal = phys_params.SURFACE_ANGLE_CORONAL_V_PROBE
    else:
        raise ValueError(f'Invalid probe {probe}')

    # First rotate unit vector ventral-dorsal vector in saggital plane
    vd = np.cos(np.radians(surface_angle_saggital))
    ap = -1. * np.sin(np.radians(surface_angle_saggital))

    # Rotate vector in coronal plane
    lr = vd * -1. * np.sin(np.radians(surface_angle_coronal))
    vd *= np.cos(np.radians(surface_angle_coronal))

    # Normalize vector
    surface_normal = np.array([lr, ap, vd])
    surface_normal /= np.linalg.norm(surface_normal)
    
    return surface_normal


def _get_brain_surface_intersept(phys_params, line_vector, line_point, probe):
    """Assumes stereotaxic coordinates."""

    # Point on the surface of the brain on the plane of the brain surface
    plane_point = np.array([
        phys_params.REFERENCE_SURFACE_POINT_LR,
        phys_params.REFERENCE_SURFACE_POINT_PA,
        phys_params.REFERENCE_SURFACE_POINT_VD,
    ])
    
    # Unit vector normal to the brain surface
    plane_normal = _get_brain_surface_normal(phys_params, probe)

    # Compute intersept with line
    intersept = _line_plane_intersept(
        plane_normal, plane_point, line_vector, line_point)
    
    return intersept


def get_np_insertion_point(phys_params, np_coords):
    """Get neuropixel insertion point stereotaxic coordinates."""

    # Get axis grid coordinates of neuropixel location
    grid_coords_axial = _grid_holes_to_grid_coords_axial(
        phys_params,
        from_posterior=np_coords.from_posterior,
        from_anterior=np_coords.from_anterior,
        from_left=np_coords.from_left,
        from_right=np_coords.from_right,
    )
    # Get neuropixel trajectory line vertex at bottom of grid
    grid_coords_0 = np.concatenate([
        grid_coords_axial,
        [phys_params.GRID_BOTTOM_ABOVE_REFERENCE_SURFACE_POINT],
    ])
    # Get neuropixel trajectory line vertex on grid-coordinate surface plane
    grid_coords_1 = np.concatenate([grid_coords_axial, [0.]])

    # Transform points from grid coordinates to stereotaxic coordinates
    stereo_coords_0 = _grid_coords_to_stereotaxic_coords(
        phys_params, grid_coords_0)
    stereo_coords_1 = _grid_coords_to_stereotaxic_coords(
        phys_params, grid_coords_1)

    # Get neuropixel trajectory line vector
    line_vector = stereo_coords_1 - stereo_coords_0
    line_vector /= np.linalg.norm(line_vector)

    # Compute neuropixel insertion point
    np_insertion_point = _get_brain_surface_intersept(
        phys_params, line_vector, stereo_coords_0, probe='np')

    return np_insertion_point


def get_depth_from_brain_surface(phys_params, probe_depth, probe):
    """Get V-probe depth from brain surface."""
    if probe == 'np':
        surface_to_cortex_thickness = (
            phys_params.SURFACE_TO_CORTEX_THICKNESS_NEUROPIXEL)
    elif probe == 'v_probe':
        surface_to_cortex_thickness = (
            phys_params.SURFACE_TO_CORTEX_THICKNESS_V_PROBE)
    else:
        raise ValueError(f'Invalid probe {probe}')

    through_dura = probe_depth.through_dura
    first_spikes = probe_depth.first_spikes
    surface = probe_depth.surface
    end = probe_depth.end
    final_depth = None
    if through_dura is None:
        if phys_params.MONKEY == 'Perle':
            if first_spikes is None:
                if end is not None and surface is not None:
                    final_depth = end - surface - surface_to_cortex_thickness
            else:
                if end is not None and first_spikes is not None:
                    final_depth = end - first_spikes
        elif phys_params.MONKEY == 'Elgar':
            if surface is None:
                if end is not None and first_spikes is not None:
                    final_depth = end - first_spikes
            else:
                final_depth = end - surface - surface_to_cortex_thickness
    else:
        if end is not None:
            final_depth = end - through_dura - phys_params.PIA_THICKNESS
    return final_depth


def get_np_tip(phys_params, np_coords, probe_depth):
    """Get neuropixel insertion point stereotaxic coordinates."""

    np_insertion_point = get_np_insertion_point(phys_params, np_coords)
    np_depth = get_depth_from_brain_surface(
        phys_params, probe_depth, probe='np')
    np_tip = np_insertion_point - np.array([0., 0., np_depth])

    return np_tip


def _get_v_probe_guide_tube_holder_endpoints_grid_coords(phys_params,
                                                         v_probe_coords):
    """In grid coordinates."""

    # Load measurements
    top_from_anterior = v_probe_coords.top_from_anterior
    top_from_posterior = v_probe_coords.top_from_posterior
    top_from_right = v_probe_coords.top_from_right
    bottom_from_anterior = v_probe_coords.bottom_from_anterior
    bottom_from_posterior = v_probe_coords.bottom_from_posterior
    bottom_from_left = v_probe_coords.bottom_from_left
    height_above_grid = v_probe_coords.height_above_grid

    if top_from_right is None:
        top_from_right = _grid_from_other_side(
            phys_params, v_probe_coords.top_from_left)

    # Get top and bottom points in grid coordinates in axial plane
    top_grid_coords_axial = _grid_holes_to_grid_coords_axial(
        phys_params,
        from_anterior=top_from_anterior,
        from_posterior=top_from_posterior,
        from_right=top_from_right,
    )
    bottom_grid_coords_axial = _grid_holes_to_grid_coords_axial(
        phys_params,
        from_anterior=bottom_from_anterior,
        from_posterior=bottom_from_posterior,
        from_left=bottom_from_left,
    )

    # Compute ventral-dorsal grid coordinates of top and bottom points
    top_vd_grid_coord = (
        phys_params.V_PROBE_GT_HOLDER_TOP_ABOVE_GRID +
        phys_params.GRID_TOP_ABOVE_REFERENCE_SURFACE_POINT + height_above_grid
    )
    axial_distance = (
        np.linalg.norm(bottom_grid_coords_axial - top_grid_coords_axial))
    bottom_top_vd_distance = np.sqrt(
        np.square(phys_params.V_PROBE_GT_HOLDER_LENGTH) -
        np.square(axial_distance)
    )
    bottom_vd_grid_coord = top_vd_grid_coord - bottom_top_vd_distance

    # Top and bottom points in grid coordinates
    top_grid_coords = np.concatenate(
        [top_grid_coords_axial, [top_vd_grid_coord]])
    bottom_grid_coords = np.concatenate(
        [bottom_grid_coords_axial, [bottom_vd_grid_coord]])

    return top_grid_coords, bottom_grid_coords


def _get_v_probe_guide_tube_holder_endpoints_stereo_coords(phys_params,
                                                           v_probe_coords):
    top_grid_coords, bottom_grid_coords = (
        _get_v_probe_guide_tube_holder_endpoints_grid_coords(
            phys_params, v_probe_coords)
    )

    top_stereo_coords = _grid_coords_to_stereotaxic_coords(
        phys_params, top_grid_coords)
    bottom_stereo_coords = _grid_coords_to_stereotaxic_coords(
        phys_params, bottom_grid_coords)

    return top_stereo_coords, bottom_stereo_coords
    

def _get_v_probe_surface_insertion_point(phys_params, v_probe_coords):
    # Get coordinates of top and bottom guide tube holed points in stereotaxic
    # coordinates
    top_stereo_coords, bottom_stereo_coords = (
        _get_v_probe_guide_tube_holder_endpoints_stereo_coords(
            phys_params, v_probe_coords)
    )

    # Compute brain surface insertion point
    line_vector = bottom_stereo_coords - top_stereo_coords
    line_vector /= np.linalg.norm(line_vector)
    intersept = _get_brain_surface_intersept(
        phys_params, line_vector, top_stereo_coords, probe='v_probe')

    return intersept, line_vector


def get_v_probe_coordinates(phys_params, v_probe_coords, v_probe_depth):
    # Compute midpoint, tip channel, and base channel in stereotaxic coordinates

    depth = get_depth_from_brain_surface(
        phys_params, v_probe_depth, probe='v_probe')
    depth_first_channel = depth - V_PROBE_PARAMS.V_PROBE_TIP_TO_FIRST_CHANNEL
    depth_last_channel = depth - V_PROBE_PARAMS.V_PROBE_TIP_TO_LAST_CHANNEL
    depth_midpoint = depth - V_PROBE_PARAMS.V_PROBE_TIP_TO_MIDPOINT
    
    intersept, line_vector = _get_v_probe_surface_insertion_point(
        phys_params, v_probe_coords)
    line_vector /= np.linalg.norm(line_vector)

    first_channel = intersept + line_vector * depth_first_channel
    last_channel = intersept + line_vector * depth_last_channel
    midpoint = intersept + line_vector * depth_midpoint

    outputs = dict(
        first_channel=first_channel,
        last_channel=last_channel,
        midpoint=midpoint,
    )

    return outputs
