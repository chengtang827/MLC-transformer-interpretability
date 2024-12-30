"""Functions for computing probe coordinates.

Grid coordinates: Origin is reference  point of surface of brain under middle
grid hole (middle in both left-right and anterior-posterior). The planes of both
neuropixel and V-probe recording areas pass through this point.

Generally L-R axis is x (index 0), P-A axis is y (index 1), D-V axis is z (index
2).

Stereotactic coordinate are w.r.t. ear bar zero and level so that the bottom of
the eye orbit is at the same z as ear bar zero. Ray between posterior and
anterior commissure is parallel with this line.

All grid holes are 1-indexed.
"""

import numpy as np

class Params():

    MONKEY = 'Perle'

    ################################################################################
    ####  GRID PROPERTIES
    ################################################################################

    # Number of grid holes per side of the square grid
    NUM_GRID_HOLES = 17
    # Distance between grid holes
    GRID_HOLE_SPACING = 1.
    # Thickness (height) of grid
    GRID_THICKNESS = 12.5

    ################################################################################
    ####  GRID ORIENTATION WITH RESPECT TO STEREOTAXIC COORDINATES
    ################################################################################

    # Grid angle in axial plane, counterclockwise in degrees, view from top
    GRID_ANGLE_AXIAL = 3.9
    # Grid angle in saggital plane, counterclockwise in degrees, view from right
    GRID_ANGLE_SAGGITAL = 2.2
    # Grid angle in coronal plane, counterclockwise in degrees, view from back
    GRID_ANGLE_CORONAL = 0.1

    ################################################################################
    ####  TISSUE MEASUREMENTS
    ################################################################################

    # Thickness of pia
    PIA_THICKNESS = 0.300
    # Thickness between surface of granulation tissue and cortex
    SURFACE_TO_CORTEX_THICKNESS_NEUROPIXEL = 1.4
    # Thickness between surface of granulation tissue and cortex along V-probe
    # trajectory
    SURFACE_TO_CORTEX_THICKNESS_V_PROBE = 3.9

    ################################################################################
    ####  SURFACE ORIENTATION WITH RESPECT TO STEREOTAXIC COORDINATES
    ################################################################################

    # Surface angle in saggital plane, counterclockwise in degrees, view from right
    SURFACE_ANGLE_SAGGITAL_NEUROPIXEL = -12.8
    # Surface angle in coronal plane, counterclockwise in degrees, view from right
    SURFACE_ANGLE_CORONAL_NEUROPIXEL = 0.

    # Surface angle in saggital plane, counterclockwise in degrees, view from right
    SURFACE_ANGLE_SAGGITAL_V_PROBE = -14.1
    # V-probe surface angle in coronal plane, counterclockwise in degrees, view from
    # back
    SURFACE_ANGLE_CORONAL_V_PROBE = 5.1

    ################################################################################
    ####  STEREOTAXIC LOCATION OF POINT ON SURFACE UNDER POSTERIOR MEDIAL GRID HOLE
    ################################################################################

    # Left-right coordinate of point
    REFERENCE_SURFACE_POINT_LR = -0.9
    # Posterior-anterior coordinate of point
    REFERENCE_SURFACE_POINT_PA = 25.0
    # Ventral-dorsal coordinate of point
    REFERENCE_SURFACE_POINT_VD = 37.6
    # Distance of reference surface point below top of grid
    GRID_TOP_ABOVE_REFERENCE_SURFACE_POINT = 27.6
    # Distance of reference surface point below bottom of grid
    GRID_BOTTOM_ABOVE_REFERENCE_SURFACE_POINT = (
        GRID_TOP_ABOVE_REFERENCE_SURFACE_POINT - GRID_THICKNESS
    )

    ################################################################################
    ####  V-PROBE GUIDE TUBE HOLDER MEASUREMENTS
    ################################################################################

    # Length of the V-probe guide tube holder
    V_PROBE_GT_HOLDER_LENGTH = 25.1
    # How far above the top of the grid is the V-probe guide tube holder
    V_PROBE_GT_HOLDER_TOP_ABOVE_GRID = 1.8

