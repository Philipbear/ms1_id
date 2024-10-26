import argparse
from bin.lcms._reverse_matching import prepare_ms2_lib
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare MS2 Library for Matching')

    # Required arguments
    parser.add_argument('--ms2db', type=str, required=True,
                        help='Path to MS2 database file in MSP format')

    # Optional matching parameters
    parser.add_argument('--mz_tol', type=float, default=0.05,
                        help='m/z tolerance for peak matching (default: 0.05)')
    parser.add_argument('--peak_scale_k', type=float, default=None,
                        help='Peak scale factor. Set to None for no scaling (default: None)')
    parser.add_argument('--peak_intensity_power', type=float, default=0.5,
                        help='Peak intensity power. Use 0.5 for square root transformation (default: 0.5)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Check if input file exists
    if not os.path.exists(args.ms2db):
        raise FileNotFoundError(f"MS2 database file not found: {args.ms2db}")

    prepare_ms2_lib(
        ms2db=args.ms2db,
        mz_tol=args.mz_tol,
        peak_scale_k=args.peak_scale_k,
        peak_intensity_power=args.peak_intensity_power
    )


if __name__ == '__main__':
    main()
