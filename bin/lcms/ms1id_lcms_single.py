"""
create a workflow for MS1_ID using masscube backend.
for single file mode.
"""

import os

from masscube.feature_grouping import annotate_isotope
from masscube.params import Params, find_ms_info
from masscube.raw_data_utils import MSData

from _annotate_adduct import annotate_adduct
from _annotate_ms2 import annotate_rois
from _calculate_ppc import calc_all_ppc
from _export import write_single_file
from _group_ppc import generate_pseudo_ms1
from _reverse_matching import ms1_id_annotation

# default parameters
orbitrap_mass_detect_int_tol = 10000
tof_mass_detect_int_tol = 500


def main_workflow_single(file_path,
                         msms_library_path,
                         ms1_id=True, ms2_id=False,
                         mz_tol_ms1=0.01, mz_tol_ms2=0.015, mass_detect_int_tol=None,
                         peak_cor_rt_tol=0.015,
                         min_ppc=0.8, roi_min_length=3,
                         ms1id_score_cutoff=0.7, ms1id_min_matched_peak=6,
                         ms1id_min_prec_int_in_ms1=1e5, ms1id_max_prec_rel_int_in_other_ms2=0.01,
                         ms2id_score_cutoff=0.7, ms2id_min_matched_peak=6,
                         plot_bpc=False, out_dir=None):
    """
    Untargeted feature detection from a single file (.mzML or .mzXML).
    """

    ms_type, ion_mode, centroid = find_ms_info(file_path)

    if not centroid:
        raise ValueError("The file is not centroided.")

    # init a new config object
    config = init_config_single(ms_type, ion_mode, msms_library_path,
                                mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
                                mass_detect_int_tol=mass_detect_int_tol)

    # create a MSData object
    d = MSData()

    # read raw data
    print('Reading raw data...')
    d.read_raw_data(file_path, config)

    # detect region of interests (ROIs)
    print('Detecting ROIs...')
    d.find_rois()

    # cut ROIs
    d.cut_rois()

    # label short ROIs, find the best MS2, and sort ROIs by m/z
    print('Summarizing ROIs...')
    d.summarize_roi()

    # annotate isotopes, adducts
    print('Annotating isotopes and adducts...')
    annotate_isotope(d)
    annotate_adduct(d)

    # generate pseudo ms1 spec for ms1_id
    pseudo_ms1_spectra = []
    if ms1_id and config.msms_library is not None:
        print("MS1 ID annotation...")

        # calc peak-peak correlations for feature groups and output
        print('Calculating peak-peak correlations...')
        ppc_matrix = calc_all_ppc(d, rt_tol=peak_cor_rt_tol, roi_min_length=roi_min_length, min_ppc=min_ppc,
                                  save=False)

        # generate pseudo ms1 spec, for ms1_id
        print('Generating pseudo MS1 spectra...')
        pseudo_ms1_spectra = generate_pseudo_ms1(d, ppc_matrix,
                                                 min_ppc=min_ppc, roi_min_length=roi_min_length,
                                                 min_cluster_size=ms1id_min_matched_peak)

        # perform rev cos search
        print('Performing MS1 ID annotation...')
        pseudo_ms1_spectra = ms1_id_annotation(pseudo_ms1_spectra, config.msms_library, mz_tol=mz_tol_ms1,
                                               ion_mode=ion_mode,
                                               min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                                               max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                                               score_cutoff=ms1id_score_cutoff, min_matched_peak=ms1id_min_matched_peak)

        # # write out raw ms1 id results
        # write_ms1_id_results(pseudo_ms1_spectra, save=True, out_dir=os.path.dirname(file_path))

    # annotate MS2 spectra
    if ms2_id and config.msms_library is not None:
        print("Annotating MS2 spectra...")
        annotate_rois(d, ms2id_score_cutoff=ms2id_score_cutoff,
                      ms2id_min_matched_peak=ms2id_min_matched_peak,
                      ion_mode=ion_mode)

    if plot_bpc:
        bpc_path = os.path.splitext(file_path)[0] + "_bpc.png"
        d.plot_bpc(label_name=True, output_dir=bpc_path)

    # output single file to a tsv file, in the same directory as the raw file
    if out_dir is None:
        out_dir = os.path.dirname(file_path)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0] + "_feature_table.tsv")
    write_single_file(d, pseudo_ms1_spectra, ion_mode, out_path)

    return d


def init_config_single(ms_type, ion_mode, msms_library_path,
                       mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                       mass_detect_int_tol=None):
    # init
    config = Params()

    if mass_detect_int_tol is not None:
        config.int_tol = mass_detect_int_tol
    else:
        if ms_type == "orbitrap":
            config.int_tol = orbitrap_mass_detect_int_tol
        elif ms_type == "tof":
            config.int_tol = tof_mass_detect_int_tol
        else:
            config.int_tol = 10000

    ##########################
    # The project
    config.project_dir = None  # Project directory, character string
    config.sample_names = None  # Absolute paths to the raw files, without extension, list of character strings
    config.sample_groups = None  # Sample groups, list of character strings
    config.sample_group_num = None  # Number of sample groups, integer
    config.sample_dir = None  # Directory for the sample information, character string
    config.single_file_dir = None  # Directory for the single file output, character string
    config.annotation_dir = None  # Directory for the annotation output, character string
    config.chromatogram_dir = None  # Directory for the chromatogram output, character string
    # config.network_dir = None             # Directory for the network output, character string
    config.statistics_dir = None  # Directory for the statistical analysis output, character string

    # MS data acquisition
    config.rt_range = [0.0, 1000.0]  # RT range in minutes, list of two floats
    config.ion_mode = ion_mode  # Ionization mode, "positive" or "negative", character string

    # Feature detection
    config.mz_tol_ms1 = mz_tol_ms1  # m/z tolerance for MS1, default is 0.01
    config.mz_tol_ms2 = mz_tol_ms2  # m/z tolerance for MS2, default is 0.015
    # config.int_tol = 30000  # Intensity tolerance, default is 30000 for Orbitrap and 1000 for other instruments, integer
    config.roi_gap = 30  # Gap within a feature, default is 30 (i.e. 30 consecutive scans without signal), integer
    config.ppr = 0.8  # Peak-peak correlation threshold for feature grouping, default is 0.7

    # Parameters for feature alignment
    config.align_mz_tol = 0.01  # m/z tolerance for MS1, default is 0.01
    config.align_rt_tol = 0.2  # RT tolerance, default is 0.2
    config.run_rt_correction = True  # Whether to perform RT correction, default is True
    config.min_scan_num_for_alignment = 5  # Minimum scan number a feature to be aligned, default is 6

    # Parameters for feature annotation
    config.msms_library = msms_library_path  # Path to the MS/MS library (.msp or .pickle), character string
    config.ms2_sim_tol = 0.7  # MS2 similarity tolerance, default is 0.7

    # Parameters for normalization
    config.run_normalization = False  # Whether to normalize the data, default is False
    config.normalization_method = "pqn"  # Normalization method, default is "pqn" (probabilistic quotient normalization), character string

    # Parameters for output
    config.output_single_file = False  # Whether to output the processed individual files to a csv file
    config.output_aligned_file = False  # Whether to output aligned features to a csv file

    # Statistical analysis
    config.run_statistics = False  # Whether to perform statistical analysis

    # # Network analysis
    # config.run_network = False            # Whether to perform network analysis

    # Visualization
    config.plot_bpc = False  # Whether to plot base peak chromatogram
    config.plot_ms2 = False  # Whether to plot mirror plots for MS2 matching

    return config


if __name__ == "__main__":
    import time

    start = time.time()

    main_workflow_single(
        file_path='/Users/shipei/Documents/projects/ms1_id/data/trial_data/single/Standards_p_1ugmL_glycocholic.mzXML',
        msms_library_path='../../data/gnps.pkl',
        ms1_id=True, ms2_id=False,
        mz_tol_ms1=0.01, mz_tol_ms2=0.015,
        mass_detect_int_tol=30000,
        peak_cor_rt_tol=0.025,
        min_ppc=0.9, roi_min_length=4,
        ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
        ms1id_min_prec_int_in_ms1=1e5,
        ms1id_max_prec_rel_int_in_other_ms2=0.01,
        ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
        plot_bpc=False)

    end = time.time()
    print("Time elapsed: ", end - start)