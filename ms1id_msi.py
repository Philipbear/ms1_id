import argparse
from bin.msi.main_msi import ms1id_single_file_batch
import os


def verify_paths(paths):
    """Verify that all paths exist"""
    if paths is None:
        return []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
    return paths


def parse_arguments():
    parser = argparse.ArgumentParser(description='MS1 ID MSI Analysis')

    # Required argument
    parser.add_argument('--project_dir', type=str, required=True,
                        help='Project directory containing imzML & ibd files')

    # Library paths
    parser.add_argument('--libs', type=str, nargs='*', default=None,
                        help='Optional: One or more paths to library files (.pkl). Paths with spaces should be quoted.')

    # Processing mode parameters
    parser.add_argument('--n_processes', type=int, default=None,
                        help='Number of processes to use (default: None, use all available cores)')
    parser.add_argument('--mass_detect_int_tol', type=float, default=None,
                        help='Mass detection intensity tolerance (default: None, automatically determined)')
    parser.add_argument('--centroided', action='store_true', default=False,
                        help='Whether the data is centroided (default: False, centroiding will be performed)')

    # Noise detection parameters
    parser.add_argument('--noise_detection', type=str, default='moving_average',
                        choices=['moving_average', 'percentile'],
                        help='Noise detection method (default: moving_average)')
    parser.add_argument('--sn_factor', type=float, default=3.0,
                        help='Signal-to-noise factor for noise removal (default: 3.0)')

    # Data processing parameters
    parser.add_argument('--mz_bin_size', type=float, default=0.01,
                        help='m/z bin size (default: 0.01)')
    parser.add_argument('--min_overlap', type=int, default=10,
                        help='Minimum overlap between ion images (default: 10)')
    parser.add_argument('--min_correlation', type=float, default=0.85,
                        help='Minimum correlation between spectra (default: 0.85)')
    parser.add_argument('--max_cor_depth', type=int, default=1,
                        help='Maximum correlation depth for spatial correlation (default: 1)')

    # Library search parameters
    parser.add_argument('--library_search_mztol', type=float, default=0.05,
                        help='Library search m/z tolerance (default: 0.05)')
    parser.add_argument('--ms1id_score_cutoff', type=float, default=0.7,
                        help='MS1 ID matching score cutoff (default: 0.7)')
    parser.add_argument('--ms1id_min_matched_peak', type=int, default=4,
                        help='MS1 ID minimum matched peaks (default: 4)')
    parser.add_argument('--ms1id_min_spec_usage', type=float, default=0.05,
                        help='MS1 ID minimum spectrum usage (default: 0.05)')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Check if project directory exists
    if not os.path.exists(args.project_dir):
        raise FileNotFoundError(f"Project directory not found: {args.project_dir}")

    # Verify library paths
    library_paths = verify_paths(args.libs)

    # Run the analysis
    ms1id_single_file_batch(
        file_dir=args.project_dir,
        library_path=library_paths,
        n_processes=args.n_processes,
        mass_detect_int_tol=args.mass_detect_int_tol,
        noise_detection=args.noise_detection,
        sn_factor=args.sn_factor,
        centroided=args.centroided,
        mz_bin_size=args.mz_bin_size,
        min_overlap=args.min_overlap,
        min_correlation=args.min_correlation,
        max_cor_depth=args.max_cor_depth,
        library_search_mztol=args.library_search_mztol,
        ms1id_score_cutoff=args.ms1id_score_cutoff,
        ms1id_min_matched_peak=args.ms1id_min_matched_peak,
        ms1id_min_spec_usage=args.ms1id_min_spec_usage
    )


if __name__ == '__main__':
    main()
