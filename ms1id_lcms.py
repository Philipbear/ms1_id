import argparse
from bin.lcms.main_lcms import ms1id_batch_mode
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
    parser = argparse.ArgumentParser(description='MS1 ID LC-MS Analysis')

    # Required argument
    parser.add_argument('--project_dir', type=str, required=True,
                        help='Path to the project directory')

    # Library and sample directory arguments
    parser.add_argument('--ms1_id_libs', type=str, nargs='*', default=None,
                        help='Optional: One or more paths to MS1 ID library files (.pkl). Paths with spaces should be quoted.')
    parser.add_argument('--ms2_id_libs', type=str, nargs='*', default=None,
                        help='Optional: One or more paths to MS2 ID library files (.pkl). Paths with spaces should be quoted.')
    parser.add_argument('--sample_dir', type=str, default='data',
                        help='Directory containing mzML/mzXML files (default: data)')

    # Processing mode flags
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                        help='Run in parallel mode (default: True)')
    parser.add_argument('--ms1_id', action='store_true', default=True,
                        help='Perform MS1-based identification (default: True)')
    parser.add_argument('--ms2_id', action='store_true', default=False,
                        help='Perform MS2-based identification (default: False)')
    parser.add_argument('--run_rt_correction', action='store_true', default=True,
                        help='Run retention time correction (default: True)')
    parser.add_argument('--run_normalization', action='store_true', default=True,
                        help='Run normalization (default: True)')

    # Processing parameters
    parser.add_argument('--cpu_ratio', type=float, default=0.9,
                        help='CPU usage ratio (default: 0.9)')
    parser.add_argument('--ms1_tol', type=float, default=0.01,
                        help='MS1 tolerance (default: 0.01)')
    parser.add_argument('--ms2_tol', type=float, default=0.02,
                        help='MS2 tolerance (default: 0.02)')
    parser.add_argument('--mass_detect_int_tol', type=float, default=3e5,
                        help='Mass detection intensity tolerance (default: 3e5)')

    # Alignment parameters
    parser.add_argument('--align_mz_tol', type=float, default=0.01,
                        help='Feature alignment m/z tolerance (default: 0.01)')
    parser.add_argument('--align_rt_tol', type=float, default=0.2,
                        help='Feature alignment retention time tolerance in minutes (default: 0.2)')
    parser.add_argument('--alignment_drop_by_fill_pct_ratio', type=float, default=0.1,
                        help='Alignment drop by fill percentage ratio (default: 0.1)')

    # Peak correlation parameters
    parser.add_argument('--peak_cor_rt_tol', type=float, default=0.025,
                        help='Peak-peak correlation retention time tolerance in minutes (default: 0.025)')
    parser.add_argument('--min_ppc', type=float, default=0.80,
                        help='Minimum peak-peak correlation to form a feature group (default: 0.80)')
    parser.add_argument('--roi_min_length', type=int, default=5,
                        help='ROI minimum length for a feature (default: 5)')

    # Library search parameters
    parser.add_argument('--library_search_mztol', type=float, default=0.05,
                        help='Library search m/z tolerance (default: 0.05)')

    # MS1 identification parameters
    parser.add_argument('--ms1id_score_cutoff', type=float, default=0.7,
                        help='MS1 ID matching score cutoff (default: 0.7)')
    parser.add_argument('--ms1id_min_matched_peak', type=int, default=4,
                        help='MS1 ID minimum matched peaks (default: 4)')
    parser.add_argument('--ms1id_min_spec_usage', type=float, default=0.20,
                        help='MS1 ID minimum spectrum usage (default: 0.20)')
    parser.add_argument('--ms1id_max_prec_rel_int_in_other_ms2', type=float, default=0.01,
                        help='MS1 ID: maximum allowed precursor relative intensity in other MS2 (default: 0.01)')

    # MS2 identification parameters
    parser.add_argument('--ms2id_score_cutoff', type=float, default=0.7,
                        help='MS2 ID score cutoff (default: 0.7)')
    parser.add_argument('--ms2id_min_matched_peak', type=int, default=4,
                        help='MS2 ID minimum matched peaks (default: 4)')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Check if project path exists
    if not os.path.exists(args.project_dir):
        raise FileNotFoundError(f"Project directory not found: {args.project_dir}")

    # Verify library paths
    ms1_libs = verify_paths(args.ms1_id_libs)
    ms2_libs = verify_paths(args.ms2_id_libs)

    # If no libraries are provided, set corresponding ID flags to False
    if not ms1_libs:
        args.ms1_id = False
    if not ms2_libs:
        args.ms2_id = False

    # Run the analysis
    ms1id_batch_mode(
        project_path=args.project_dir,
        ms1id_library_path=ms1_libs,
        ms2id_library_path=ms2_libs,
        sample_dir=args.sample_dir,
        parallel=args.parallel,
        ms1_id=args.ms1_id,
        ms2_id=args.ms2_id,
        cpu_ratio=args.cpu_ratio,
        run_rt_correction=args.run_rt_correction,
        run_normalization=args.run_normalization,
        ms1_tol=args.ms1_tol,
        ms2_tol=args.ms2_tol,
        mass_detect_int_tol=args.mass_detect_int_tol,
        align_mz_tol=args.align_mz_tol,
        align_rt_tol=args.align_rt_tol,
        alignment_drop_by_fill_pct_ratio=args.alignment_drop_by_fill_pct_ratio,
        peak_cor_rt_tol=args.peak_cor_rt_tol,
        min_ppc=args.min_ppc,
        roi_min_length=args.roi_min_length,
        library_search_mztol=args.library_search_mztol,
        ms1id_score_cutoff=args.ms1id_score_cutoff,
        ms1id_min_matched_peak=args.ms1id_min_matched_peak,
        ms1id_min_spec_usage=args.ms1id_min_spec_usage,
        ms1id_max_prec_rel_int_in_other_ms2=args.ms1id_max_prec_rel_int_in_other_ms2,
        ms2id_score_cutoff=args.ms2id_score_cutoff,
        ms2id_min_matched_peak=args.ms2id_min_matched_peak
    )


if __name__ == '__main__':
    main()
