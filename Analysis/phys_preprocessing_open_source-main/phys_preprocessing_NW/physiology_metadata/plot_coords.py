"""Generate and plot stereotactic coordinates."""

from matplotlib import pyplot as plt
from moog_demos import gif_writer as gif_writer_lib
import numpy as np
import os
import phys_metadata_perle
import phys_metadata_elgar

_MONKEY_NAME = 'elgar'
_WRITE_GIF = False


def _scatter3(points, colors, title=''):
    """General 3D scatterplot function."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
    )
    ax.set_xlabel('Left-Right')
    ax.set_ylabel('Posterior-Anterior')
    ax.set_zlabel('Ventral-Dorsal')
    ax.set_title(title)
    ax.set_box_aspect((
        np.ptp(points[:, 0]),
        np.ptp(points[:, 1]),
        np.ptp(points[:, 2]),
    ))

    return fig, ax


def _write_to_gif(fig, ax, gif_name):
    gif_file = os.path.join(os.getcwd(), 'gifs', gif_name)
    gif_writer = gif_writer_lib.GifWriter(gif_file=gif_file, fps=8)

    # Rotate plot and save frames to gif_writer
    frames = []
    for angle in range(0, 360, 5):
        ax.view_init(30, angle)
        # Must scatter something to trigger re-rendering, so scatter zero-size
        # point.
        ax.scatter([0.], [30.], [25.], s=0)
        fig.canvas.draw()
        plt.pause(.001)
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)

    frames = np.array(frames)
    frames_white = frames == 255
    frames_x_white = np.prod(frames_white, axis=(0, 2, 3))
    frames_y_white = np.prod(frames_white, axis=(0, 1, 3))
    x_min = np.min(np.argwhere(frames_x_white == 0)) - 1
    x_max = np.max(np.argwhere(frames_x_white == 0)) + 1
    y_min = np.min(np.argwhere(frames_y_white == 0)) - 1
    y_max = np.max(np.argwhere(frames_y_white == 0)) + 1
    cropped_frames = frames[:, x_min: x_max, y_min: y_max]

    for frame in cropped_frames:
        gif_writer.add(frame)
    gif_writer.close()


def main():

    if _MONKEY_NAME == 'perle':
        sessions = phys_metadata_perle.PERLE
    elif _MONKEY_NAME == 'elgar':
        sessions = phys_metadata_elgar.ELGAR
    else:
        raise ValueError(f'Invalid _MONKEY_NAME {_MONKEY_NAME}')

    str(sessions['2022-05-18'])

    import pdb; pdb.set_trace()

    ############################################################################
    #### EXPLORE NEUROPIXEL RECORDING SITES
    ############################################################################

    np_coords = [
        (date, session.np_0_coords)
        for date, session in sessions.items()
        if session.np_0_coords is not None
    ]

    # Extract coordinates
    np_stereo_coords = np.array([x[1] for x in np_coords])
    np_depths = np.array([
        sessions[x[0]].np_depth_from_brain_surface
        for x in np_coords
    ])

    _scatter3(np_stereo_coords, np_depths, title='Neuropixel Recording Sites')

    ############################################################################
    #### PLOT V-PROBE RECORDING SITES
    ############################################################################

    def _plot_v_probe_coords(probe_number):
        v_probe_coords = [
            (date, getattr(session, f'v_probe_{probe_number}_insertion_coords'))
            for date, session in sessions.items()
            if (
                getattr(session, f'v_probe_{probe_number}_insertion_coords')
                is not None
            )
        ]

        # Extract coordinates
        v_probe_midpoints = np.array([x[1]['midpoint'] for x in v_probe_coords])
        v_probe_first_channels = np.array(
            [x[1]['first_channel'] for x in v_probe_coords])
        v_probe_last_channels = np.array(
            [x[1]['last_channel'] for x in v_probe_coords])
        v_probe_depths = np.array([
            sessions[x[0]].v_probe_depth_from_brain_surface
            for x in v_probe_coords
        ])

        # Concatenate midpoints, first channels, and last channels for plotting
        v_probe_points_to_plot = np.concatenate([
            v_probe_first_channels, v_probe_midpoints, v_probe_last_channels
        ], axis=0)
        v_probe_depths_to_plot = np.tile(v_probe_depths, 3)

        if len(v_probe_depths_to_plot) > 0:
            _scatter3(
                v_probe_points_to_plot,
                v_probe_depths_to_plot,
                title=f'V-Probe {probe_number} Recording Sites',
            )
            return v_probe_points_to_plot, v_probe_depths_to_plot
        else:
            return None, None

    v_probe_0_points_to_plot, v_probe_0_depths_to_plot = _plot_v_probe_coords(0)
    v_probe_1_points_to_plot, v_probe_1_depths_to_plot = _plot_v_probe_coords(1)

    ############################################################################
    #### PLOT ALL RECORDING SITES
    ############################################################################
    
    if v_probe_1_points_to_plot is None:
        points_to_plot = np.concatenate(
            [v_probe_0_points_to_plot, np_stereo_coords])
        depths_to_plot = np.concatenate([v_probe_0_depths_to_plot, np_depths])
    else:
        points_to_plot = np.concatenate([
            v_probe_0_points_to_plot, v_probe_1_points_to_plot, np_stereo_coords
        ])
        depths_to_plot = np.concatenate([
            v_probe_0_depths_to_plot, v_probe_1_depths_to_plot, np_depths
        ])

    fig, ax = _scatter3(
        points_to_plot,
        depths_to_plot,
        title='All Recording Sites',
    )
    if _WRITE_GIF:
        ax.view_init(30, 110)
        _write_to_gif(fig, ax, 'all_recording_sites_' + _MONKEY_NAME + '.gif')
    print('Done')

    plt.show()


if __name__ == '__main__':
    main()