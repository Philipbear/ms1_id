import numpy as np


def get_msi_features(mz_array, intensity_array, mz_ppm_tol=5):
    """
    Generate unique m/z values from a list of m/z values considering a ppm tolerance.
    M/z values are first sorted by their intensities in descending order.
    Each unique m/z value is calculated as the average of all m/z values in its group.

    Parameters:
    -----------
    mz_array : array-like
        List or array of m/z values
    intensity_array : array-like
        List or array of intensity values corresponding to the m/z values
    mz_ppm_tol : float
        PPM tolerance for grouping m/z values (default: 5)

    Returns:
    --------
    np.ndarray
        Array of unique m/z values (centroids)
    """
    # Convert inputs to numpy arrays
    mz_array = np.array(mz_array)
    intensity_array = np.array(intensity_array)

    # Sort m/z values by intensity in descending order
    sort_idx = np.argsort(intensity_array)[::-1]
    sorted_mzs = mz_array[sort_idx]

    unique_mzs = []
    current_group = [sorted_mzs[0]]

    # Process each m/z value
    for i in range(1, len(sorted_mzs)):
        current_mz = sorted_mzs[i]
        group_mean = np.mean(current_group)

        # Calculate ppm difference
        ppm_diff = abs(current_mz - group_mean) / group_mean * 1e6

        # If within tolerance, add to current group
        if ppm_diff <= mz_ppm_tol:
            current_group.append(current_mz)
        # If outside tolerance, finalize current group and start new one
        else:
            unique_mzs.append(np.mean(current_group))
            current_group = [current_mz]

    # Add last group
    unique_mzs.append(np.mean(current_group))

    return unique_mzs


def assign_spectrum_to_feature(idx, mz_array, intensity_array, feature_mzs):
    """
    Assign m/z values to features based on a list of unique m/z values.
    """
    # Initialize dictionary to store feature intensities
    feature_intensities = {mz: 0 for mz in feature_mzs}

    # Iterate through m/z values and intensities
    for mz, intensity in zip(mz_array, intensity_array):
        # Find the closest feature m/z value
        closest_feature_mz = min(feature_mzs, key=lambda x: abs(x - mz))
        # Add intensity to feature
        feature_intensities[closest_feature_mz] += intensity

    return idx, feature_intensities


# Example usage
if __name__ == "__main__":
    # Example data
    mz_example = [
        100.001, 100.002, 100.003, 100.004, 100.005,  # Should be grouped together
        200.001, 200.002,  # Should be grouped together
        300.001, 300.009  # Should be separate features
    ]

    intensity_example = [
        500, 550, 450, 480, 520,  # Intensities for first group
        1000, 980,  # Intensities for second group
        100, 90  # Intensities for third group
    ]

    # Generate unique features with 15 ppm tolerance
    unique_features = get_msi_features(mz_example, intensity_example, mz_ppm_tol=15)
    print("Original m/z values:", mz_example)
    print("Original intensities:", intensity_example)
    print("Unique m/z features:", unique_features)