import pickle
from collections import defaultdict
import os
import numpy as np
import pyimzml.ImzMLParser as imzml


def process_ms_imaging_data(imzml_file, ibd_file, mass_detect_int_tol=500.0,
                            bin_size=0.005, save=False):
    parser = imzml.ImzMLParser(imzml_file)

    mz_intensity_dict = defaultdict(lambda: defaultdict(float))
    coordinates = []

    for idx, (x, y, z) in enumerate(parser.coordinates):
        mz, intensity = parser.getspectrum(idx)

        # Filter intensities and bin m/z values in one step
        mask = intensity > mass_detect_int_tol
        binned_mz = np.round(mz[mask] / bin_size) * bin_size
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

    if save:
        file_name = os.path.basename(imzml_file).replace('.imzML', '')
        save_dir = os.path.dirname(imzml_file)

        mz_values_path = os.path.join(save_dir, f'{file_name}_mz_values.npy')
        intensity_matrix_path = os.path.join(save_dir, f'{file_name}_intensity_matrix.npy')
        coordinates_path = os.path.join(save_dir, f'{file_name}_coordinates.pkl')

        np.save(mz_values_path, mz_values)
        np.save(intensity_matrix_path, intensity_matrix)

        with open(coordinates_path, 'wb') as f:
            pickle.dump(coordinates, f)

    return mz_values, intensity_matrix, coordinates


def analyze_intensity_distribution(intensity_matrix):
    # Flatten the intensity matrix to get all intensity values
    all_intensities = intensity_matrix.flatten()

    # Remove zero intensities
    non_zero_intensities = all_intensities[all_intensities > 0]

    # Calculate percentiles
    percentiles = [1, 2, 5, 50, 95, 98, 99]
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


if __name__ == '__main__':
    imzml_file = '../../imaging/bottom_control_1.imzML'
    mz_values, intensity_matrix, coordinates = process_ms_imaging_data(imzml_file, imzml_file.replace('.imzML', '.ibd'))
    intensity_stats = analyze_intensity_distribution(intensity_matrix)
    print_intensity_stats(intensity_stats)
