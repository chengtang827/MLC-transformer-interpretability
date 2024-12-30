"""Compute and write SpikeGLX imec sample rate."""

import json
import os
import sys


def _get_imec_sample_rate(imec_meta_file):
    with open(imec_meta_file) as f:
        imec_meta_lines = f.readlines()
    imec_meta = {
        x.split('=', 1)[0]: x.split('=', 1)[1][:-1]
        for x in imec_meta_lines
    }
    sample_rate = float(imec_meta['imSampRate'])
    
    return sample_rate


def main():
    """Compute and write sample rate for spikeglx imec data.
    
    Args:
        metadata_file: String. Full path to spikeglx metadata file.
        write_dir: String. Full path to directory to write sample rate.
    """

    metadata_file = sys.argv[1]
    print(f'metadata_file: {metadata_file}')
    write_dir = sys.argv[2]
    print(f'write_dir: {write_dir}')

    # Get sample_rate
    sample_rate = _get_imec_sample_rate(metadata_file)

    # Write sample_rate
    sample_rate_path = os.path.join(write_dir, 'sample_rate')
    print(
        f'Writing sample_rate {sample_rate} to '
        f'{sample_rate_path}'
    )
    json.dump(sample_rate, open(sample_rate_path, 'w'))
    print('Done extracting sample rate.')

    return


if __name__ == "__main__":
    main()
