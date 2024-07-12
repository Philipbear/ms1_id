"""
create a workflow for MS1_ID using masscube backend.
"""

import multiprocessing
import os
import pickle
from copy import deepcopy

from masscube.feature_grouping import annotate_isotope
from masscube.alignment import feature_alignment, gap_filling, output_feature_table
from masscube.annotation import feature_annotation, annotate_rois
from masscube.feature_table_utils import convert_features_to_df
from masscube.normalization import sample_normalization
from masscube.params import Params, find_ms_info
from masscube.raw_data_utils import MSData

from _annotate_adduct import annotate_adduct
from _calculate_ppc import calc_all_ppc
from _group_ppc_aligned_feature import generate_pseudo_ms1
from _revcos import ms1_id_annotation, write_ms1_id_results


def main_workflow(project_path=None, msms_library_path=None, sample_dir='data',
                  ms1_id=True, ms2_id=False,
                  batch_size=100, cpu_ratio=0.8,
                  run_rt_correction=True, run_normalization=False,
                  mz_tol_ms1=0.01, mz_tol_ms2=0.015, mass_detect_int_tol=None,
                  align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                  peak_cor_rt_tol=0.1, hdbscan_prob_cutoff=0.2,
                  ms1id_score_cutoff=0.8, ms1id_min_matched_peak=6):
    """
    Main workflow for MS1_ID.
    :param project_path: path to the project directory
    :param msms_library_path: path to the MS/MS library
    :param ms1_id: whether to perform MS1 ID
    :param ms2_id: whether to perform MS2 ID
    :param batch_size: batch size for multiprocessing
    :param cpu_ratio: CPU ratio for multiprocessing
    :param sample_dir: directory where the raw files are stored
    :param run_rt_correction: whether to perform RT correction
    :param run_normalization: whether to perform normalization
    :param mz_tol_ms1: m/z tolerance for MS1
    :param mz_tol_ms2: m/z tolerance for MS2
    :param mass_detect_int_tol: intensity tolerance for mass detection
    :param align_mz_tol: alignment m/z tolerance
    :param align_rt_tol: alignment RT tolerance
    :param alignment_drop_by_fill_pct_ratio: alignment drop by fill percentage ratio
    :param peak_cor_rt_tol: peak correlation RT tolerance
    :param hdbscan_prob_cutoff: HDBSCAN probability cutoff
    :param ms1id_score_cutoff: ms1 ID score cutoff
    :param ms1id_min_matched_peak: ms1 ID min matched peak
    :return:
    """

    # init a new config object
    config = init_config(project_path, msms_library_path,
                         sample_dir=sample_dir, mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
                         mass_detect_int_tol=mass_detect_int_tol,
                         align_mz_tol=align_mz_tol, align_rt_tol=align_rt_tol,
                         run_rt_correction=run_rt_correction, run_normalization=run_normalization)

    with open(os.path.join(config.project_dir, "project.mc"), "wb") as f:
        pickle.dump(config, f)

    raw_file_names = os.listdir(config.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    raw_file_names = [f for f in raw_file_names if not f.startswith(".")]  # for Mac OS
    total_file_num = len(raw_file_names)

    # skip the files that have been processed
    txt_files = os.listdir(config.single_file_dir)
    txt_files = [f.split(".")[0] for f in txt_files if f.lower().endswith(".txt")]
    txt_files = [f for f in txt_files if not f.startswith(".")]  # for Mac OS
    raw_file_names = [f for f in raw_file_names if f.split(".")[0] not in txt_files]
    raw_file_names = [os.path.join(config.sample_dir, f) for f in raw_file_names]

    print("\t{} raw file are found, {} files to be processed.".format(total_file_num, len(raw_file_names)))

    # process files by multiprocessing, each batch contains 100 files by default (tunable in batch_size)
    print("Processing individual files for feature detection, evaluation, and grouping...")
    workers = int(multiprocessing.cpu_count() * cpu_ratio)
    print("\tA total of {} CPU cores are detected, {} cores are used.".format(multiprocessing.cpu_count(), workers))
    for i in range(0, len(raw_file_names), batch_size):
        if len(raw_file_names) - i < batch_size:
            print("Processing files from " + str(i) + " to " + str(len(raw_file_names)))
        else:
            print("Processing files from " + str(i) + " to " + str(i + batch_size))
        p = multiprocessing.Pool(workers)
        p.starmap(feature_detection, [(f, config) for f in raw_file_names[i:i + batch_size]])
        p.close()
        p.join()

    # feature alignment
    print("Aligning features...")
    features = feature_alignment(config.single_file_dir, config,
                                 drop_by_fill_pct_ratio=alignment_drop_by_fill_pct_ratio)

    # gap filling
    print("Filling gaps...")
    features = gap_filling(features, config)

    if ms1_id and config.msms_library is not None and os.path.exists(config.msms_library):
        print("MS1 ID annotation...")
        # generate pseudo ms1 spec from aligned table, for ms1_id
        pseudo_ms1_spectra = generate_pseudo_ms1(config, features, peak_cor_rt_tol=peak_cor_rt_tol,
                                                 hdbscan_prob_cutoff=hdbscan_prob_cutoff)

        # perform rev cos search
        pseudo_ms1_spectra = ms1_id_annotation(pseudo_ms1_spectra, config.msms_library, mz_tol=mz_tol_ms1,
                                               precursor_in_spec=True,
                                               score_cutoff=ms1id_score_cutoff, min_matched_peak=ms1id_min_matched_peak)

        # write out ms1 id results
        output_path = os.path.join(config.project_dir, "aligned_feature_table_ms1_id.tsv")
        write_ms1_id_results(pseudo_ms1_spectra, output_path)

    # annotation (using MS2 library)
    print("Annotating features (MS2)...")
    if ms2_id and config.msms_library is not None and os.path.exists(config.msms_library):
        features = feature_annotation(features, config)
        print("\tMS2 annotation is completed.")

    feature_table = convert_features_to_df(features, config.sample_names)

    # normalization
    if config.run_normalization:
        print("Running normalization...")
        output_path = os.path.join(config.project_dir, "aligned_feature_table_before_normalization.txt")
        output_feature_table(feature_table, output_path)
        # feature_table_before_normalization = deepcopy(feature_table)
        feature_table = sample_normalization(feature_table, config.individual_sample_groups,
                                             config.normalization_method)

    # output feature table
    output_path = os.path.join(config.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    print("The workflow is completed.")


def init_config(path=None, msms_library_path=None,
                sample_dir='data', mz_tol_ms1=0.01, mz_tol_ms2=0.015, mass_detect_int_tol=None,
                align_mz_tol=0.01, align_rt_tol=0.2, run_rt_correction=True, run_normalization=False):
    # init
    config = Params()
    # obtain the working directory
    if path is not None:
        config.project_dir = path
    else:
        config.project_dir = os.getcwd()

    config.sample_dir = os.path.join(config.project_dir, sample_dir)
    config.single_file_dir = os.path.join(config.project_dir, "single_files")

    # check if the required files are prepared
    if not os.path.exists(config.sample_dir) or len(os.listdir(config.sample_dir)) == 0:
        raise ValueError("No raw MS data is found in the project directory.")

    # create the output directories if not exist
    if not os.path.exists(config.single_file_dir):
        os.makedirs(config.single_file_dir)

    # determine the type of MS and ion mode
    file_names = os.listdir(config.sample_dir)
    file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    file_name = os.path.join(config.sample_dir, file_names[0])
    ms_type, ion_mode, _ = find_ms_info(file_name)

    if mass_detect_int_tol is not None:
        config.int_tol = mass_detect_int_tol
    else:
        if ms_type == "orbitrap":
            config.int_tol = 5000
        elif ms_type == "tof":
            config.int_tol = 500

    ##########################
    # The project
    # config.project_dir = None  # Project directory, character string
    config.sample_names = [f.split(".")[0] for f in os.listdir(config.sample_dir)]
    config.sample_groups = ["sample"] * len(config.sample_names)
    config.individual_sample_groups = ["sample"] * len(config.sample_names)
    config.sample_group_num = 1

    # config.sample_dir = None  # Directory for the sample information, character string
    # config.single_file_dir = None  # Directory for the single file output, character string
    config.annotation_dir = None  # Directory for the annotation output, character string
    config.chromatogram_dir = None  # Directory for the chromatogram output, character string
    config.network_dir = None  # Directory for the network output, character string
    config.statistics_dir = None  # Directory for the statistical analysis output, character string

    # MS data acquisition
    config.rt_range = [0.0, 1000.0]  # RT range in minutes, list of two floats
    config.ion_mode = ion_mode  # Ionization mode, "positive" or "negative", character string

    # Feature detection
    config.mz_tol_ms1 = mz_tol_ms1  # m/z tolerance for MS1, default is 0.01
    config.mz_tol_ms2 = mz_tol_ms2  # m/z tolerance for MS2, default is 0.015
    # config.int_tol = 30000  # Intensity tolerance, default is 30000 for Orbitrap and 1000 for other instruments, integer
    config.roi_gap = 30  # Gap within a feature, default is 30 (i.e. 30 consecutive scans without signal), integer
    config.ppr = 0.7  # Peak peak correlation threshold for feature grouping, default is 0.7

    # Parameters for feature alignment
    config.align_mz_tol = align_mz_tol  # m/z tolerance for MS1, default is 0.01
    config.align_rt_tol = align_rt_tol  # RT tolerance, default is 0.2
    config.run_rt_correction = run_rt_correction  # Whether to perform RT correction, default is True
    config.min_scan_num_for_alignment = 6  # Minimum scan number a feature to be aligned, default is 6

    # Parameters for feature annotation
    config.msms_library = msms_library_path  # Path to the MS/MS library (.msp or .pickle), character string
    config.ppr = 0.7  # Peak-peak correlation threshold for feature grouping, default is 0.7
    config.ms2_sim_tol = 0.7  # MS2 similarity tolerance, default is 0.7

    # Parameters for normalization
    config.run_normalization = run_normalization  # Whether to normalize the data, default is False
    config.normalization_method = "pqn"  # Normalization method, default is "pqn" (probabilistic quotient normalization), character string

    # Parameters for output
    config.output_single_file = True  # Whether to output the processed individual files to a csv file
    config.output_aligned_file = True  # Whether to output aligned features to a csv file

    # Statistical analysis
    config.run_statistics = False  # Whether to perform statistical analysis

    # Network analysis
    config.run_network = False  # Whether to perform network analysis

    # Visualization
    config.plot_bpc = False  # Whether to plot base peak chromatogram
    config.plot_ms2 = False  # Whether to plot mirror plots for MS2 matching

    return config


def feature_detection(file_name, params=None, cal_g_score=True, cal_a_score=True,
                      anno_isotope=True, anno_adduct=True, anno_in_source_fragment=False,
                      annotation=False, ms2_library_path=None, cut_roi=True):
    """
    Untargeted feature detection from a single file (.mzML or .mzXML).

    Returns: An MSData object containing the processed data.
    """

    # try:
    # except Exception as e:
    #     print("Error: " + str(e))
    #     return None

    # create a MSData object
    d = MSData()

    # set parameters
    ms_type, ion_mode, centroid = find_ms_info(file_name)
    if not centroid:
        print("File: " + file_name + " is not centroided and skipped.")
        return None

    # if params is None, use the default parameters
    if params is None:
        params = Params()
        params.set_default(ms_type, ion_mode)

    if ms2_library_path is not None:
        params.msms_library = ms2_library_path

    # read raw data
    d.read_raw_data(file_name, params)

    # detect region of interests (ROIs)
    d.find_rois()

    # cut ROIs
    if cut_roi:
        d.cut_rois()

    # label short ROIs, find the best MS2, and sort ROIs by m/z
    d.summarize_roi(cal_g_score=cal_g_score, cal_a_score=cal_a_score)

    if anno_isotope:
        annotate_isotope(d, mz_tol=0.015, rt_tol=0.1)
    # if anno_in_source_fragment:
    #     annotate_in_source_fragment(d)
    if anno_adduct:
        annotate_adduct(d, mz_tol=0.01, rt_tol=0.05)

    # calc peak-peak correlations for feature groups and output
    calc_all_ppc(d, rt_tol=0.1)

    # output single file to a txt file
    d.output_single_file()

    return d


if __name__ == "__main__":

    # params = init_config(path='/Users/shipei/Documents/test_data/mzML')
    # feature_detection(file_name='/Users/shipei/Documents/test_data/mzML/data/000001527_RG8_01_6306.mzML', params=params)

    main_workflow(project_path='/Users/shipei/Documents/test_data/mzML',
                  msms_library_path='/Users/shipei/Documents/projects/ms1_id/data/MassBank_NIST.pkl',
                  sample_dir='data',
                  ms1_id=True, ms2_id=False,
                  batch_size=100, cpu_ratio=0.8,
                  run_rt_correction=True, run_normalization=False,
                  mz_tol_ms1=0.01, mz_tol_ms2=0.015, mass_detect_int_tol=30000,
                  align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                  peak_cor_rt_tol=0.1, hdbscan_prob_cutoff=0.2,
                  ms1id_score_cutoff=0.7, ms1id_min_matched_peak=6)

    # from masscube.workflows import untargeted_metabolomics_workflow
    # untargeted_metabolomics_workflow(path='/Users/shipei/Documents/test_data/mzML')
