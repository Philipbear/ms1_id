import os
import argparse
from ms1_id.lcms.reverse_matching import prepare_ms2_lib
from ms1_id.lcms.main_lcms import ms1id_batch_mode as ms1id_lcms
from ms1_id.msi.main_msi import ms1id_single_file_batch as ms1id_msi
from ms1_id.annotate import annotate_pseudo_ms2_spec


def index_library(args):
    """Function to handle library indexing"""

    # Check if input file exists
    if not os.path.exists(args.ms2db):
        raise FileNotFoundError(f"MS2 database file not found: {args.ms2db}")

    print(f"Indexing library from: {args.ms2db}")
    # print(f"Using m/z tolerance: {args.mz_tol}")
    print(f"Peak scaling factor: {args.peak_scale_k}")
    print(f"Peak intensity power: {args.peak_intensity_power}")

    # Call your actual indexing function here
    prepare_ms2_lib(
        ms2db=args.ms2db,
        # mz_tol=args.mz_tol,
        peak_scale_k=args.peak_scale_k,
        peak_intensity_power=args.peak_intensity_power
    )


def annotate_spectra(args):
    """Function to handle spectra annotation"""

    # Check if input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Verify library paths
    lib_ls = _verify_paths(args.libs)
    if not lib_ls:
        raise ValueError("No valid library paths provided")

    print(f"Annotating spectra from: {args.input_file}")
    print(f"Using libraries: {args.libs}")
    print(f"m/z tolerance: {args.mz_tol}")
    print(f"Ion mode: {args.ion_mode}")
    print(f"Score cutoff: {args.min_score}")
    print(f"Minimum matched peak: {args.min_matched_peak}")
    print(f"Minimum spectral usage: {args.min_spec_usage}")
    if args.output_folder is None:
        print("No output folder specified. Results will saved in the same folder as the input file.")
    else:
        print(f"Output folder: {args.output_folder}")

    annotate_pseudo_ms2_spec(
        ms2_file_path=args.input_file,
        library_ls=lib_ls,
        mz_tol=args.mz_tol,
        ion_mode=args.ion_mode,
        score_cutoff=args.min_score,
        min_matched_peak=args.min_matched_peak,
        min_spec_usage=args.min_spec_usage,
        save_dir=args.output_folder
    )


def run_lcms(args):

    # Check if project path exists
    if not os.path.exists(args.project_dir):
        raise FileNotFoundError(f"Project directory not found: {args.project_dir}")

    # Verify library paths
    ms1_libs = _verify_paths(args.ms1_id_libs)

    print(f"Running LC-MS data analysis in project directory: {args.project_dir}")
    print(f"Sample directory: {args.sample_dir}")

    if ms1_libs:
        print("MS1 ID enabled")
        print(f"Using MS1 ID libraries: {ms1_libs}")
    else:
        print("MS1 ID disabled, as no MS1 ID libraries are provided")

    if args.ms2_id_lib:
        print("MS2 ID enabled")
        print(f"Using MS2 ID libraries: {args.ms2_id_lib}")
    else:
        print("MS2 ID disabled, as no MS2 ID libraries are provided")

    print(f"Parallel processing: {args.parallel}")
    print(f"CPU ratio: {args.cpu_ratio}")

    # Run the analysis
    ms1id_lcms(
        project_path=args.project_dir,
        ms1id_library_path=ms1_libs,
        ms2id_library_path=args.ms2_id_lib,
        sample_dir=args.sample_dir,
        parallel=args.parallel,
        ms1_id=True if ms1_libs else False,
        ms2_id=True if args.ms2_id_lib else False,
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


def run_msi(args):

    # Check if project directory exists
    if not os.path.exists(args.project_dir):
        raise FileNotFoundError(f"Project directory not found: {args.project_dir}")

    # Verify library paths
    library_paths = _verify_paths(args.libs)

    print(f"Running MS imaging data analysis in project directory: {args.project_dir}")
    print(f"Using libraries: {library_paths}")
    print(f"Number of cores used: {args.n_cores}")

    # Run the analysis
    ms1id_msi(
        file_dir=args.project_dir,
        library_path=library_paths,
        n_processes=args.n_cores,
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


def _verify_paths(paths):
    """Verify that all paths exist"""
    if paths is None:
        return []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description='Annotate LC-MS1 data, MS imaging data or pseudo MS/MS spectra using reference MS/MS libraries')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    ####################
    # Subcommand for indexing library
    index_parser = subparsers.add_parser('index', help='Index MS/MS library from a msp or mgf file')
    index_parser.add_argument('--ms2db', type=str, required=True,
                        help='Path to MS/MS database file in mgf or msp format')
    # index_parser.add_argument('--mz_tol', type=float, default=0.05,
    #                     help='m/z tolerance for peak matching (default: 0.05)')
    index_parser.add_argument('--peak_scale_k', type=float, default=None,
                        help='Peak scaling factor. Set to None for no scaling (default: None)')
    index_parser.add_argument('--peak_intensity_power', type=float, default=1.0,
                        help='Peak intensity power. Use 0.5 for square root transformation (default: 1.0, no transformation)')

    ####################
    # Subcommand for annotation
    annotate_parser = subparsers.add_parser('annotate', help='Annotate pseudo MS/MS spectra in mgf format')
    annotate_parser.add_argument('--input_file', '-i', type=str, required=True,
                        help='Path to the MGF file containing pseudo MS/MS spectra')
    annotate_parser.add_argument('--libs', '-l', type=str, required=True, nargs='*', default=None,
                        help='One or more paths to indexed library files (.pkl). Paths with spaces should be quoted.')
    annotate_parser.add_argument('--output_folder', '-o', type=str, default=None,
                        help='Output folder for annotated results')
    annotate_parser.add_argument('--mz_tol', type=float, default=0.05,
                        help='m/z tolerance for peak matching (default: 0.05)')
    annotate_parser.add_argument('--ion_mode', type=str, default=None,
                        help='Ion mode, "positive" or "negative" (default: None)')
    annotate_parser.add_argument('--min_score', type=float, default=0.7,
                        help='Minimum score for matching (default: 0.7)')
    annotate_parser.add_argument('--min_matched_peak', type=int, default=3,
                        help='Minimum number of matched peaks (default: 3)')
    annotate_parser.add_argument('--min_spec_usage', type=float, default=0.20,
                        help='Minimum spectral usage (default: 0.20)')

    ####################
    # Subcommand for LC-MS data
    lcms_parser = subparsers.add_parser('lcms', help='Annotate LC-MS spectra in mzML or mzXML format')
    lcms_parser.add_argument('--project_dir', type=str, required=True,
                             help='Path to the project directory')
    lcms_parser.add_argument('--sample_dir', type=str, default='data',
                             help='Directory containing mzML or mzXML files (default: data)')
    lcms_parser.add_argument('--ms1_id_libs', type=str, nargs='*', default=None,
                        help='Optional: One or more paths to MS1 ID library files (.pkl). Paths with spaces should be quoted.')
    lcms_parser.add_argument('--ms2_id_lib', type=str, default=None,
                        help='Optional: Path to MS2 ID library file (.pkl).')

    lcms_parser.add_argument('--parallel', '-p', action='store_true', default=True,
                             help='Run in parallel mode (default: True)')
    lcms_parser.add_argument('--no_parallel', action='store_false', dest='parallel',
                             help='Disable parallel mode')

    lcms_parser.add_argument('--run_rt_correction', action='store_true', default=True,
                             help='Run retention time correction (default: True)')
    lcms_parser.add_argument('--no_run_rt_correction', action='store_false', dest='run_rt_correction',
                             help='Skip retention time correction')

    lcms_parser.add_argument('--run_normalization', action='store_true', default=True,
                             help='Run normalization (default: True)')
    lcms_parser.add_argument('--no_run_normalization', action='store_false', dest='run_normalization',
                             help='Skip normalization')

    lcms_parser.add_argument('--cpu_ratio', type=float, default=0.9,
                        help='CPU usage ratio (default: 0.9)')
    lcms_parser.add_argument('--ms1_tol', type=float, default=0.01,
                        help='MS1 tolerance (default: 0.01)')
    lcms_parser.add_argument('--ms2_tol', type=float, default=0.02,
                        help='MS2 tolerance (default: 0.02)')
    lcms_parser.add_argument('--mass_detect_int_tol', type=float, default=None,
                        help='Mass detection intensity tolerance (default: None, 2e5 for Orbitraps, 5e2 for Q-TOFs, 2e5 for others)')
    lcms_parser.add_argument('--align_mz_tol', type=float, default=0.01,
                        help='Feature alignment m/z tolerance (default: 0.01)')
    lcms_parser.add_argument('--align_rt_tol', type=float, default=0.2,
                        help='Feature alignment retention time tolerance in minutes (default: 0.2)')
    lcms_parser.add_argument('--alignment_drop_by_fill_pct_ratio', type=float, default=0.1,
                        help='Alignment drop by fill percentage ratio (default: 0.1)')
    lcms_parser.add_argument('--peak_cor_rt_tol', type=float, default=0.025,
                        help='Peak-peak correlation retention time tolerance in minutes (default: 0.025)')
    lcms_parser.add_argument('--min_ppc', type=float, default=0.80,
                        help='Minimum peak-peak correlation to form a feature group (default: 0.80)')
    lcms_parser.add_argument('--roi_min_length', type=int, default=5,
                        help='ROI minimum length for a feature (default: 5)')
    lcms_parser.add_argument('--library_search_mztol', type=float, default=0.05,
                        help='Library search m/z tolerance (default: 0.05)')
    lcms_parser.add_argument('--ms1id_score_cutoff', type=float, default=0.7,
                        help='MS1 ID matching score cutoff (default: 0.7)')
    lcms_parser.add_argument('--ms1id_min_matched_peak', type=int, default=3,
                        help='MS1 ID minimum matched peaks (default: 3)')
    lcms_parser.add_argument('--ms1id_min_spec_usage', type=float, default=0.20,
                        help='MS1 ID minimum spectrum usage (default: 0.20)')
    lcms_parser.add_argument('--ms1id_max_prec_rel_int_in_other_ms2', type=float, default=0.01,
                        help='MS1 ID: maximum allowed precursor relative intensity in other MS2 (default: 0.01)')
    lcms_parser.add_argument('--ms2id_score_cutoff', type=float, default=0.7,
                        help='MS2 ID score cutoff (default: 0.7)')
    lcms_parser.add_argument('--ms2id_min_matched_peak', type=int, default=3,
                        help='MS2 ID minimum matched peaks (default: 3)')

    ####################
    # Subcommand for MSI data
    msi_parser = subparsers.add_parser('msi', help='Annotate MS imaging data in imzML & ibd format')
    msi_parser.add_argument('--project_dir', type=str, required=True,
                        help='Project directory containing imzML & ibd files')
    msi_parser.add_argument('--libs', '-l', type=str, nargs='*', default=None,
                        help='One or more paths to library files (.pkl). Paths with spaces should be quoted.')
    msi_parser.add_argument('--n_cores', type=int, default=None,
                        help='Number of cores to use (default: None, use all available cores)')
    msi_parser.add_argument('--mass_detect_int_tol', type=float, default=None,
                        help='Mass detection intensity tolerance (default: None, automatically determined)')
    msi_parser.add_argument('--centroided', action='store_true', default=False,
                        help='Whether the data is centroided (default: False, centroiding will be performed)')
    msi_parser.add_argument('--noise_detection', type=str, default='moving_average',
                        choices=['moving_average', 'percentile'],
                        help='Noise detection method (default: moving_average)')
    msi_parser.add_argument('--sn_factor', type=float, default=3.0,
                        help='Signal-to-noise factor for noise removal (default: 3.0)')
    msi_parser.add_argument('--mz_bin_size', type=float, default=0.01,
                        help='m/z bin size (default: 0.01)')
    msi_parser.add_argument('--min_overlap', type=int, default=10,
                        help='Minimum overlap between ion images (default: 10)')
    msi_parser.add_argument('--min_correlation', type=float, default=0.85,
                        help='Minimum correlation between spectra (default: 0.85)')
    msi_parser.add_argument('--max_cor_depth', type=int, default=1,
                        help='Maximum correlation depth for spatial correlation (default: 1)')
    msi_parser.add_argument('--library_search_mztol', type=float, default=0.05,
                        help='Library search m/z tolerance (default: 0.05)')
    msi_parser.add_argument('--ms1id_score_cutoff', type=float, default=0.7,
                        help='MS1 ID matching score cutoff (default: 0.7)')
    msi_parser.add_argument('--ms1id_min_matched_peak', type=int, default=3,
                        help='MS1 ID minimum matched peaks (default: 3)')
    msi_parser.add_argument('--ms1id_min_spec_usage', type=float, default=0.05,
                        help='MS1 ID minimum spectrum usage (default: 0.05)')

    args = parser.parse_args()

    # Execute appropriate function based on subcommand
    if args.command == 'index':
        index_library(args)
    elif args.command == 'annotate':
        annotate_spectra(args)
    elif args.command == 'lcms':
        run_lcms(args)
    elif args.command == 'msi':
        run_msi(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()