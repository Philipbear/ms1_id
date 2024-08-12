import pickle
from collections import defaultdict
import os
import numpy as np
import pyimzml.ImzMLParser as imzml


def process_ms_imaging_data(imzml_file, ibd_file, mass_detect_int_tol=None, max_mz=None,
                            mz_bin_size=0.005, save=False, save_dir=None):
    parser = imzml.ImzMLParser(imzml_file)

    all_intensities = []
    # First pass: collect all intensities if mass_detect_int_tol is None
    if mass_detect_int_tol is None:
        for idx, (x, y, z) in enumerate(parser.coordinates):
            mz, intensity = parser.getspectrum(idx)
            all_intensities.extend(intensity)

        all_intensities = np.array(all_intensities)
        non_zero_intensities = all_intensities[all_intensities > 0.0]
        # Calculate intensity value for mass detection
        mass_detect_int_tol = max(np.min(non_zero_intensities) * 3, np.percentile(non_zero_intensities, 5))

    # check if result files exist
    if save_dir is not None:
        mz_values_path = os.path.join(save_dir, 'mz_values.npy')
        intensity_matrix_path = os.path.join(save_dir, 'intensity_matrix.npy')
        coordinates_path = os.path.join(save_dir, 'coordinates.pkl')

        if os.path.exists(mz_values_path) and os.path.exists(intensity_matrix_path) and os.path.exists(
                coordinates_path):
            mz_values = np.load(mz_values_path)
            intensity_matrix = np.load(intensity_matrix_path)
            with open(coordinates_path, 'rb') as f:
                coordinates = pickle.load(f)

            return mz_values, intensity_matrix, coordinates, parser.polarity, mass_detect_int_tol

    mz_intensity_dict = defaultdict(lambda: defaultdict(float))
    coordinates = []

    for idx, (x, y, z) in enumerate(parser.coordinates):
        mz, intensity = parser.getspectrum(idx)

        # Filter intensities and bin m/z values in one step
        mask = (intensity > mass_detect_int_tol) & ((max_mz is None) or (mz <= max_mz))
        binned_mz = np.round(mz[mask] / mz_bin_size) * mz_bin_size
        filtered_intensity = intensity[mask]

        # Update mz_intensity_dict
        for m, i in zip(binned_mz, filtered_intensity):
            mz_intensity_dict[m][idx] = i

        coordinates.append((x, y))

    # Convert to numpy arrays
    mz_values = np.array(sorted(mz_intensity_dict.keys()))
    intensity_matrix = np.array([
        [mz_intensity_dict[mz].get(idx, 0.0) for idx in range(len(coordinates))]
        for mz in mz_values
    ])

    if save and save_dir is not None:
        mz_values_path = os.path.join(save_dir, 'mz_values.npy')
        intensity_matrix_path = os.path.join(save_dir, 'intensity_matrix.npy')
        coordinates_path = os.path.join(save_dir, 'coordinates.pkl')
        np.save(mz_values_path, mz_values)
        np.save(intensity_matrix_path, intensity_matrix)

        with open(coordinates_path, 'wb') as f:
            pickle.dump(coordinates, f)

    return mz_values, intensity_matrix, coordinates, parser.polarity, mass_detect_int_tol


def analyze_intensity_distribution(intensity_matrix):
    # Flatten the intensity matrix to get all intensity values
    all_intensities = intensity_matrix.flatten()

    # Remove zero intensities
    non_zero_intensities = all_intensities[all_intensities > 0]

    # Calculate percentiles
    percentiles = [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99]
    intensity_percentiles = np.percentile(non_zero_intensities, percentiles)

    # Create a dictionary of percentiles and their corresponding intensity values
    percentile_dict = dict(zip(percentiles, intensity_percentiles))

    # Calculate some additional statistics
    mean_intensity = np.mean(non_zero_intensities)
    median_intensity = np.median(non_zero_intensities)
    min_intensity = np.min(non_zero_intensities)
    max_intensity = np.max(non_zero_intensities)

    return {
        "percentiles": percentile_dict,
        "mean": mean_intensity,
        "median": median_intensity,
        "min": min_intensity,
        "max": max_intensity,
        "total_intensities": len(all_intensities),
        "non_zero_intensities": len(non_zero_intensities)
    }


def print_intensity_stats(stats):
    print("Intensity Distribution Analysis:")
    print(f"Total number of intensity values: {stats['total_intensities']}")
    print(f"Number of non-zero intensity values: {stats['non_zero_intensities']}")
    print("\nPercentiles:")
    for percentile, value in stats['percentiles'].items():
        print(f"{percentile}th percentile: {value:.2f}")
    print(f"\nMean intensity: {stats['mean']:.2f}")
    print(f"Median intensity: {stats['median']:.2f}")
    print(f"Minimum non-zero intensity: {stats['min']:.2f}")
    print(f"Maximum intensity: {stats['max']:.2f}")


def create_intensity_histogram(intensity_matrix, bins=1000, percentile_cutoff=95):
    import matplotlib.pyplot as plt
    # Flatten the intensity matrix to get all intensity values
    all_intensities = intensity_matrix.flatten()

    # Remove zero intensities
    non_zero_intensities = all_intensities[all_intensities > 0]

    # Calculate the cutoff value for the x-axis (90th percentile by default)
    x_max = np.percentile(non_zero_intensities, percentile_cutoff)

    # Create the histogram
    plt.figure(figsize=(12, 6))

    # Use range parameter to limit x-axis
    counts, bins, _ = plt.hist(non_zero_intensities, bins=bins, range=(0, x_max), edgecolor='black')

    plt.title(f'Histogram of Non-Zero Intensity Values (First {percentile_cutoff}%)')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    # Add vertical lines for percentiles
    percentiles = [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
    for percentile, color in zip(percentiles, colors):
        value = np.percentile(non_zero_intensities, percentile)
        if value <= x_max:
            plt.axvline(value, color=color, linestyle='dashed', linewidth=1)
            plt.text(value, plt.gca().get_ylim()[1], f'{percentile}th',
                     rotation=90, va='top', ha='right', color=color)

    # Add text box with statistics
    stats_text = f"Total values: {len(all_intensities)}\n"
    stats_text += f"Non-zero values: {len(non_zero_intensities)}\n"
    stats_text += f"Max shown: {x_max:.2f}"
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    imzml_file = '../../imaging/mouse_body/wb xenograft in situ metabolomics test - rms_corrected.imzML'
    mz_values, intensity_matrix, coordinates, ion_mode, _ = process_ms_imaging_data(imzml_file,
                                                                                    imzml_file.replace('.imzML',
                                                                                                       '.ibd'),
                                                                                    mass_detect_int_tol=None)

    intensity_stats = analyze_intensity_distribution(intensity_matrix)
    print_intensity_stats(intensity_stats)
    create_intensity_histogram(intensity_matrix, bins=1000, percentile_cutoff=50)
