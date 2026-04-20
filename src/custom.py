import numpy as np
import os
import glob
import h5py
import pandas as pd

def nearest_index_with_nan(coord, values):
    """
    Return nearest index in coord for each value.

    Valid values get the nearest sample index.
    NaN values stay NaN.
    """
    coord = np.asarray(coord).squeeze()
    values = np.asarray(values).squeeze()

    idx = np.full(values.shape, np.nan, dtype=float)

    good = np.isfinite(values)
    if not np.any(good):
        return idx

    vals_good = values[good]

    ins = np.searchsorted(coord, vals_good)
    ins = np.clip(ins, 1, len(coord) - 1)

    left = coord[ins - 1]
    right = coord[ins]

    use_left = np.abs(vals_good - left) <= np.abs(right - vals_good)
    ins[use_left] = ins[use_left] - 1

    idx[good] = ins.astype(float)
    return idx


def _get_freq_scalar_from_radar_param(radar_group, reference_name):
    """
    Return a single unique scalar from radar_group[reference_name].

    Works whether the dataset stores:
    1. numeric values directly, or
    2. HDF5 references to datasets containing the values.
    """
    raw = np.array(radar_group[reference_name][:]).squeeze()

    if raw.size == 0:
        raise RuntimeError(f"The field '{reference_name}' is empty")

    flat = np.atleast_1d(raw).flatten()

    values = []

    for item in flat:
        if isinstance(item, h5py.Reference):
            values.append(radar_group.file[item][()])
        else:
            values.append(item)

    values_array = np.array(values).squeeze()
    unique_values = np.unique(values_array)

    if len(unique_values) == 1:
        return unique_values[0]
    else:
        raise RuntimeError(
            f"The field '{reference_name}' contains multiple unique values: {unique_values}"
        )

def load_radar_freq_params_from_first_file(directory_path):
    file_list = sorted(glob.glob(os.path.join(directory_path, "*.mat")),
                       key=lambda x: os.path.basename(x))

    if not file_list:
        raise FileNotFoundError(f"No .mat files found in {directory_path}")

    file_path = file_list[0]
    print(f"Reading frequency parameters from: {file_path}")

    with h5py.File(file_path, "r") as mat_contents:
        radar_group = mat_contents["param_qlook"]["radar"]["wfs"]

        radar_freq_params = {
            "fs": [_get_freq_scalar_from_radar_param(radar_group, "fs_raw")],
            "f0": [_get_freq_scalar_from_radar_param(radar_group, "f0")],
            "f1": [_get_freq_scalar_from_radar_param(radar_group, "f1")],
            "fc": [_get_freq_scalar_from_radar_param(radar_group, "fc")],
        }

    return pd.DataFrame(radar_freq_params)
