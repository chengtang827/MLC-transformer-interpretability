"""Physiology notes utilities."""

import collections
import json
import numpy as np
import probe_coordinates as probe_coords_lib

Neuropixel_Coords = collections.namedtuple(
    'Neuropixel_Coords',
    [
        'from_posterior',
        'from_anterior',
        'from_left',
        'from_right',
    ],
    defaults=[None, None, None, None],
)
V_Probe_Coords = collections.namedtuple(
    'V_Probe_Coords',
    [
        'top_from_anterior',
        'top_from_posterior',
        'top_from_left',
        'top_from_right',
        'bottom_from_anterior',
        'bottom_from_posterior',
        'bottom_from_left',
        'height_above_grid',
    ],
    defaults=[None, None, None, None, None, None, None, 0.],
)
Probe_Depth = collections.namedtuple(
    'Probe_Depth',
    [
        'surface',
        'through_dura',
        'first_spikes',
        'end',
    ],
    defaults=[None, None, None, None],
)


def _serialize(x):
    """Serialize a value x.
    
    This is used to serialize probe coordinate data so it is JSON-writable.
    Specifically, numpy arrays are not JSON serializable, so we must convert
    numpy arrays to lists. This function is recursive to handle nestings inside
    of lists/tuples/dictionaries.
    
    Args:
        x: Value to serialize.
    
    Returns:
        Serialized value that can be JSON dumped.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (np.int32, np.int64)):
        return int(x)
    elif isinstance(x, list):
        return [_serialize(a) for a in x]
    elif isinstance(x, tuple):
        return tuple([_serialize(a) for a in x])
    elif isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    else:
        return x


class Session():
    def __init__(self,
                 phys_params,
                 np_0_coords=None,
                 np_1_coords=None,
                 np_depth=None,
                 v_probe_0_coords=None,
                 v_probe_1_coords=None,
                 v_probe_depth=None,
                 notes=''):
        self.phys_params = phys_params
        self._np_0_coords = np_0_coords
        self._np_1_coords = np_1_coords
        self._np_depth = np_depth
        self._v_probe_0_coords = v_probe_0_coords
        self._v_probe_1_coords = v_probe_1_coords
        self._v_probe_depth = v_probe_depth
        self.notes = notes

        ########################################################################
        #### ADD PROBE STEREOTAXIC COORDINATES
        ########################################################################
        
        if self._np_depth is not None:
            self.np_depth_from_brain_surface = (
                probe_coords_lib.get_depth_from_brain_surface(
                    self.phys_params, self._np_depth, probe='np')
            )
        else:
            self.np_depth_from_brain_surface = None

        if self._np_0_coords is not None:
            self.np_0_insertion_coords = (
                probe_coords_lib.get_np_insertion_point(
                    self.phys_params, self._np_0_coords)
            )
            self.np_0_coords = (
                probe_coords_lib.get_np_tip(
                    self.phys_params, self._np_0_coords, self._np_depth)
            )
        else:
            self.np_0_insertion_coords = None
            self.np_0_coords = None

        if self._np_1_coords is not None:
            self.np_1_insertion_coords = (
                probe_coords_lib.get_np_insertion_point(
                    self.phys_params, self._np_1_coords)
            )
            self.np_1_coords = (
                probe_coords_lib.get_np_tip(
                    self.phys_params, self._np_1_coords, self._np_depth)
            )
        else:
            self.np_1_insertion_coords = None
            self.np_1_coords = None

        if self._v_probe_depth is not None:
            self.v_probe_depth_from_brain_surface = (
                probe_coords_lib.get_depth_from_brain_surface(
                    self.phys_params, self._v_probe_depth, probe='v_probe')
            )
        else:
            self.v_probe_depth_from_brain_surface = None
        
        if self._v_probe_0_coords is not None:
            self.v_probe_0_coords = (
                probe_coords_lib.get_v_probe_coordinates(
                    self.phys_params,
                    self._v_probe_0_coords,
                    self._v_probe_depth,
                )
            )
        else:
            self.v_probe_0_coords = None

        if self._v_probe_1_coords is not None:
            self.v_probe_1_coords = (
                probe_coords_lib.get_v_probe_coordinates(
                    self.phys_params,
                    self._v_probe_1_coords,
                    self._v_probe_depth,
                )
            )
        else:
            self.v_probe_1_coords = None

    def __str__(self):
        coordinate_system = (
            'Coordinate system is stereotactic coordinates, with origin at '
            'zero ear bar and pitched to the bottom of the eye orbit. '
            'Coordinate vectors are [left-right, posterior-anterior, '
            'ventral-dorsal]. Units are in millimeters.'
        )
        data = {
            'coordinate_system': coordinate_system,
            'neuropixel_depth': self.np_depth_from_brain_surface,
            'neuropixel_0_coordinates': self.np_0_coords,
            'neuropixel_1_coordinates': self.np_1_coords,
            'v_probe_depth': self.v_probe_depth_from_brain_surface,
            'v_probe_0_coordinates': self.v_probe_0_coords,
            'v_probe_1_coordinates': self.v_probe_1_coords,
            'notes': self.notes,
        }
        data_string = json.dumps(_serialize(data))
        return data_string

