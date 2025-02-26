import numpy as np
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from ms1_id.msi.centroid_data import centroid_spectrum
from pyImagingMSpec.image_measures import measure_of_chaos


def clean_msi_spec(mz, intensity, sn_factor, centroided):
    baseline = moving_average_denoising(mz, intensity, factor=sn_factor)
    mask = intensity > baseline

    # Filter mz and intensity arrays
    mz = mz[mask]
    intensity = intensity[mask]

    # Centroid the spectrum
    if not centroided:
        mz, intensity = centroid_spectrum(mz, intensity,
                                          centroid_mode='max', width_da=0.002, width_ppm=10)

    return mz, intensity


@njit
def moving_average_denoising(mz_array, intensity_array,
                             mz_window=100.0, percentage_lowest=0.05, factor=3.0):
    """
    Apply moving average algorithm to a single mass spectrum using an m/z-based window.

    :param mz_array: numpy array of m/z values
    :param intensity_array: numpy array of intensity values
    :param mz_window: size of the moving window in Da (default: 100.0)
    :param percentage_lowest: percentage of lowest points to consider in each window (default: 5%)
    :param factor: factor to multiply the mean of the lowest points (default: 5.0)
    :return: tuple of (filtered_mz_array, filtered_intensity_array)
    """
    baseline_array = []

    for i in range(len(mz_array)):
        mz_min = mz_array[i] - mz_window / 2
        mz_max = mz_array[i] + mz_window / 2

        window_intensities = intensity_array[(mz_array >= mz_min) & (mz_array <= mz_max)]

        if len(window_intensities) > 0:
            positive_intensities = window_intensities[window_intensities > 0]
            if len(positive_intensities) > 0:
                sorted_intensities = np.sort(positive_intensities)
                n_lowest = max(1, int(len(sorted_intensities) * percentage_lowest))
                lowest_n = sorted_intensities[:n_lowest]
                baseline = factor * np.mean(lowest_n)
            else:
                baseline = 0.0
        else:
            baseline = 0.0

        baseline_array.append(baseline)

    return np.array(baseline_array)


@njit
def _process_mz_group(mz_values, intensities, ppm_tol):
    """
    Numba-accelerated helper function to process a group of m/z values
    """
    n = len(mz_values)
    group_mzs = np.zeros(n)
    group_ints = np.zeros(n)

    group_mzs[0] = mz_values[0]
    group_ints[0] = intensities[0]
    group_size = 1
    current_mean = mz_values[0]

    for i in range(1, n):
        mz_diff = abs(mz_values[i] - current_mean)
        max_allowed_diff = max(current_mean * ppm_tol * 1e-6, 200 * ppm_tol * 1e-6)
        if mz_diff <= max_allowed_diff:
            group_mzs[group_size] = mz_values[i]
            group_ints[group_size] = intensities[i]
            group_size += 1
            current_mean = (group_mzs[:group_size] * group_ints[:group_size]).sum() / group_ints[:group_size].sum()
        else:
            break

    return group_mzs[:group_size], group_ints[:group_size]


def _process_chunk_get_features(chunk_data, ppm_tol, min_group_size):
    """
    Process a chunk of data and return unique m/z values
    """
    chunk_mzs, chunk_intensities = chunk_data
    i = 0
    chunk_unique_mzs = []

    while i < len(chunk_mzs):
        group_mzs, group_ints = _process_mz_group(chunk_mzs[i:], chunk_intensities[i:], ppm_tol)

        if len(group_mzs) >= min_group_size:
            chunk_unique_mzs.append(np.average(group_mzs, weights=group_ints))

        i += len(group_mzs)

    return chunk_unique_mzs


def _find_chunk_boundaries(mz_array, ppm_tol, min_chunk_size=1000):
    """
    Find safe splitting points for parallelization by identifying gaps in m/z values.
    For m/z < 200, use fixed tolerance of 200*ppm_tol/1e6.
    Ensure each chunk has at least min_chunk_size elements.

    Parameters:
    -----------
    mz_array : np.ndarray
        Sorted array of m/z values
    ppm_tol : float
        PPM tolerance for grouping m/z values
    min_chunk_size : int
        Minimum size for each chunk

    Returns:
    --------
    np.ndarray
        Indices where it's safe to split the data
    """
    # Calculate differences between consecutive m/z values
    mz_diffs = np.diff(mz_array)

    # Calculate the maximum allowed difference for each m/z value based on ppm tolerance
    # For m/z < 200, use fixed tolerance of 200*ppm_tol/1e6
    threshold_mz = 200
    max_allowed_diffs = np.where(
        mz_array[:-1] < threshold_mz,
        threshold_mz * ppm_tol * 1e-6,
        mz_array[:-1] * ppm_tol * 1e-6
    )

    # Find indices where difference is larger than allowed
    split_points = np.where(mz_diffs > max_allowed_diffs)[0] + 1

    # Add start and end points
    split_points = np.concatenate(([0], split_points, [len(mz_array)]))

    # Merge small chunks to meet minimum size requirement
    chunk_sizes = np.diff(split_points)
    merged_splits = [split_points[0]]  # Start with first point
    current_size = 0

    for i in range(len(chunk_sizes)):
        current_size += chunk_sizes[i]

        # If we've accumulated enough points or this is the last chunk
        if current_size >= min_chunk_size or i == len(chunk_sizes) - 1:
            merged_splits.append(split_points[i + 1])
            current_size = 0

    return np.array(merged_splits)


def get_msi_features(mz_array, intensity_array, mz_ppm_tol=5.0, min_group_size=10,
                     n_processes=4, min_chunk_size=10000):
    """
    Parallel processing version of get_msi_features using multiprocessing.
    Splits data at safe points where m/z differences are larger than group tolerance.
    For m/z < 200, uses fixed tolerance of 200*ppm_tol/1e6.

    Parameters:
    -----------
    mz_array : array-like
        List or array of m/z values
    intensity_array : array-like
        List or array of intensity values corresponding to the m/z values
    mz_ppm_tol : float
        PPM tolerance for grouping m/z values (default: 5)
    min_group_size : int
        Minimum number of m/z values required in a group (default: 10)
    n_processes : int
        Number of processes to use (default: 4)
    min_chunk_size : int
        Minimum size for each processing chunk (default: 10000)

    Returns:
    --------
    np.ndarray
        Array of unique m/z values (centroids) from groups with sufficient size
    """
    # Convert inputs to numpy arrays
    mz_array = np.array(mz_array)
    intensity_array = np.array(intensity_array)

    # Sort by m/z values
    sort_idx = np.argsort(mz_array)
    sorted_mzs = mz_array[sort_idx]
    sorted_intensities = intensity_array[sort_idx]

    # Find safe splitting points
    split_points = _find_chunk_boundaries(sorted_mzs, mz_ppm_tol, min_chunk_size)

    # Create chunks based on split points
    chunks = []
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        chunks.append((sorted_mzs[start:end], sorted_intensities[start:end]))

    # # Print chunk statistics
    # chunk_sizes = [len(chunk[0]) for chunk in chunks]
    # print(f"\nChunk statistics:")
    # print(f"Number of chunks: {len(chunks)}")
    # print(f"Chunk sizes - min: {min(chunk_sizes)}, max: {max(chunk_sizes)}, mean: {np.mean(chunk_sizes):.1f}\n")

    # Create partial function with fixed parameters
    process_chunk_partial = partial(_process_chunk_get_features,
                                    ppm_tol=mz_ppm_tol,
                                    min_group_size=min_group_size)

    # Process chunks in parallel with progress bar
    unique_mzs = []
    with Pool(processes=n_processes) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for chunk_result in pool.imap(process_chunk_partial, chunks):
                unique_mzs.extend(chunk_result)
                pbar.update()

    # Return empty array if no groups meet the size requirement
    if not unique_mzs:
        return np.array([])

    # Sort the unique m/z values
    return np.sort(np.array(unique_mzs))


@njit
def assign_spec_to_feature_array(mz_arr, intensity_arr, feature_mzs, mz_ppm_tol) -> np.ndarray:
    """
    Assign intensities to features based on m/z values and intensities.
    Skips assignments that exceed the PPM tolerance.

    Parameters:
    -----------
    mz_arr : np.ndarray
        Array of m/z values
    intensity_arr : np.ndarray
        Array of intensity values
    feature_mzs : np.ndarray
        Sorted array of feature m/z values
    mz_ppm_tol : float
        Mass tolerance in parts per million (PPM)

    Returns:
    --------
    np.ndarray
        Array of feature intensities
    """
    feature_intensities = np.zeros(len(feature_mzs), dtype=np.float64)
    for mz, intensity in zip(mz_arr, intensity_arr):
        idx = np.searchsorted(feature_mzs, mz)

        # Handle edge cases
        if idx == 0:
            closest_idx = 0
            closest_mz = feature_mzs[0]
        elif idx == len(feature_mzs):
            closest_idx = len(feature_mzs) - 1
            closest_mz = feature_mzs[-1]
        else:
            # Compare distances to neighbors
            if abs(feature_mzs[idx] - mz) < abs(feature_mzs[idx - 1] - mz):
                closest_idx = idx
                closest_mz = feature_mzs[idx]
            else:
                closest_idx = idx - 1
                closest_mz = feature_mzs[idx - 1]

        # Check if within PPM tolerance
        ppm_diff = abs(mz - closest_mz) / closest_mz * 1e6
        if ppm_diff <= mz_ppm_tol:
            feature_intensities[closest_idx] += intensity

    return feature_intensities


def calc_spatial_chaos_easy_version(feature_intensity_array, x_size, y_size):
    """
    Calculate the spatial chaos of a feature intensity array.
    :param feature_intensity_array: 1D array of feature intensities
    """
    im = feature_intensity_array.reshape(y_size, x_size)
    chaos = measure_of_chaos(im, 30)

    # check if nan
    if np.isnan(chaos):
        chaos = 0.0

    return chaos


def calc_spatial_chaos(feature_intensity_array, coordinates):
    """
    Calculate the spatial chaos of a feature intensity array with disordered coordinates.

    Parameters:
    -----------
    feature_intensity_array : numpy.ndarray
        1D array of feature intensities
    coordinates : list
        List of (x, y, z) coordinates corresponding to each intensity value

    Returns:
    --------
    float
        Spatial chaos score
    """
    # Extract x and y coordinates
    x_coords = np.array([coord[0] for coord in coordinates])
    y_coords = np.array([coord[1] for coord in coordinates])

    # Find min and max coordinates if not provided
    min_x = np.min(x_coords)
    min_y = np.min(y_coords)

    max_x = np.max(x_coords)
    max_y = np.max(y_coords)

    # Calculate dimensions
    x_size = max_x - min_x + 1
    y_size = max_y - min_y + 1

    # Create empty 2D array
    im = np.zeros((y_size, x_size))

    # Fill the 2D array with intensity values at their specific positions
    for i, (x, y, _) in enumerate(coordinates):
        # Adjust coordinates to start from 0
        adjusted_x = x - min_x
        adjusted_y = y - min_y

        # Place intensity value at the correct position
        im[adjusted_y, adjusted_x] = feature_intensity_array[i]

    # Calculate spatial chaos
    chaos = measure_of_chaos(im, 30)

    # Check if nan
    if np.isnan(chaos):
        chaos = 0.0

    return chaos
