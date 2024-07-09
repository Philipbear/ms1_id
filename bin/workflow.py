"""
create a workflow for MS1_ID using masscube backend.
"""

import os
import multiprocessing
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
from tqdm import tqdm

from masscube.raw_data_utils import MSData, get_start_time
from masscube.params import Params, find_ms_info
# from masscube.feature_grouping import annotate_isotope, annotate_adduct, annotate_in_source_fragment
from group_feature import annotate_isotope, calc_all_ppc, annotate_adduct
from masscube.alignment import feature_alignment, gap_filling, output_feature_table
from masscube.annotation import feature_annotation, annotate_rois, output_ms2_to_msp
from masscube.normalization import sample_normalization
from masscube.visualization import plot_ms2_matching_from_feature_table
from masscube.network import network_analysis
from masscube.stats import statistical_analysis
from masscube.feature_table_utils import calculate_fill_percentage


def main_workflow(path=None, batch_size=100, cpu_ratio=0.8):
    # init a new config object
    config = init_config(path)

    with open(os.path.join(config.project_dir, "project.mc"), "wb") as f:
        pickle.dump(config, f)

    raw_file_names = os.listdir(config.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    raw_file_names = [f for f in raw_file_names if not f.startswith(".")]  # for Mac OS
    # skip the files that have been processed
    txt_files = os.listdir(config.single_file_dir)
    txt_files = [f.split(".")[0] for f in txt_files if f.lower().endswith(".txt")]
    txt_files = [f for f in txt_files if not f.startswith(".")]  # for Mac OS
    raw_file_names = [f for f in raw_file_names if f.split(".")[0] not in txt_files]
    raw_file_names = [os.path.join(config.sample_dir, f) for f in raw_file_names]

    print("Total number of files to be processed: " + str(len(raw_file_names)))
    # process files by multiprocessing, each batch contains 100 files by default (tunable in batch_size)
    print("Processing files by multiprocessing...")
    workers = int(multiprocessing.cpu_count() * cpu_ratio)
    for i in range(0, len(raw_file_names), batch_size):
        if len(raw_file_names) - i < batch_size:
            print("Processing files from " + str(i) + " to " + str(len(raw_file_names)))
        else:
            print("Processing files from " + str(i) + " to " + str(i + batch_size))
        p = multiprocessing.Pool(workers)
        p.starmap(feature_detection, [(f, config) for f in raw_file_names[i:i + batch_size]])
        p.close()
        p.join()

    if not os.path.exists(os.path.join(config.project_dir, "aligned_feature_table_before_normalization.txt")):
        # feature alignment
        print("Aligning features...")
        feature_table = feature_alignment(config.single_file_dir, config)

        # gap filling
        print("Filling gaps...")
        feature_table = gap_filling(feature_table, config)

        # calculate fill percentage
        feature_table = calculate_fill_percentage(feature_table, params.individual_sample_groups)

        # annotation
        print("Annotating features...")
        if params.msms_library is not None and os.path.exists(params.msms_library):
            feature_annotation(feature_table, params)
        else:
            print("No MS2 library is found. Skipping annotation...")

        output_path = os.path.join(params.project_dir, "ms2.msp")
        output_ms2_to_msp(feature_table, output_path)
    else:
        feature_table = pd.read_csv(os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt"),
                                    sep="\t")

    # normalization
    if params.run_normalization:
        output_path = os.path.join(params.project_dir, "aligned_feature_table_before_normalization.txt")
        output_feature_table(feature_table, output_path)
        feature_table_before_normalization = deepcopy(feature_table)
        print("Running normalization...")
        feature_table = sample_normalization(feature_table, params.individual_sample_groups,
                                             params.normalization_method)

    # statistical analysis
    if params.run_statistics:
        print("Running statistical analysis...")
        feature_table_before_normalization = statistical_analysis(feature_table_before_normalization, params,
                                                                  before_norm=True)
        feature_table = statistical_analysis(feature_table, params)

    # output feature table
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    # network analysis
    if params.run_network:
        print("Running network analysis...This may take several minutes...")
        network_analysis(feature_table)

    # plot annoatated metabolites
    if params.plot_ms2:
        print("Plotting annotated metabolites...")
        plot_ms2_matching_from_feature_table(feature_table, params)

    # output feature table
    output_path = os.path.join(params.project_dir, "aligned_feature_table.txt")
    output_feature_table(feature_table, output_path)

    # output parameters and metadata
    params.output_parameters(os.path.join(params.project_dir, "data_processing_metadata.json"))
    print("The workflow is completed.")


def init_config(path=None, sample_dir='data'):
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

    if ms_type == "orbitrap":
        config.int_tol = 30000
    elif ms_type == "tof":
        config.int_tol = 1000

    ##########################
    # The project
    # config.project_dir = None  # Project directory, character string
    config.sample_names = [f.split(".")[0] for f in os.listdir(config.sample_dir)]  # Absolute paths to the raw files, without extension, list of character strings
    config.sample_groups = ["sample"] * len(config.sample_names)  # Sample groups, list of character strings
    config.individual_sample_groups = ["sample"] * len(config.sample_names)
    config.sample_group_num = 1  # Number of sample groups, integer
    # config.sample_dir = None  # Directory for the sample information, character string
    # config.single_file_dir = None  # Directory for the single file output, character string
    config.annotation_dir = None  # Directory for the annotation output, character string
    config.chromatogram_dir = None  # Directory for the chromatogram output, character string
    config.network_dir = None  # DirectoTry for the network output, character string
    config.statistics_dir = None  # Directory for the statistical analysis output, character string

    # MS data acquisition
    config.rt_range = [0.0, 1000.0]  # RT range in minutes, list of two floats
    config.ion_mode = ion_mode  # Ionization mode, "positive" or "negative", character string

    # Feature detection
    config.mz_tol_ms1 = 0.01  # m/z tolerance for MS1, default is 0.01
    config.mz_tol_ms2 = 0.015  # m/z tolerance for MS2, default is 0.015
    # config.int_tol = 30000  # Intensity tolerance, default is 30000 for Orbitrap and 1000 for other instruments, integer
    config.roi_gap = 30  # Gap within a feature, default is 10 (i.e. 10 consecutive scans without signal), integer
    config.min_ion_num = 10  # Minimum scan number a feature, default is 10, integer

    # Parameters for feature alignment
    config.align_mz_tol = 0.01  # m/z tolerance for MS1, default is 0.01
    config.align_rt_tol = 0.2  # RT tolerance, default is 0.2

    # Parameters for feature annotation
    config.msms_library = None  # Path to the MS/MS library (.msp or .pickle), character string
    config.ppr = 0.7  # Peak-peak correlation threshold for feature grouping, default is 0.7
    config.ms2_sim_tol = 0.7  # MS2 similarity tolerance, default is 0.7

    # Parameters for normalization
    config.run_normalization = False  # Whether to normalize the data, default is False
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

    try:
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

        # annotate isotopes, adducts, and in-source fragments
        if anno_isotope:
            annotate_isotope(d)
        # if anno_in_source_fragment:
        #     annotate_in_source_fragment(d)
        if anno_adduct:
            annotate_adduct(d)

        # calc peak-peak correlations for feature groups
        ppc_score_dict = calc_all_ppc(d)

        #########################
        # prepare peak groups and rev cos match
        # note: one roi (metabolic feature) can appear in multiple peak groups
        #########################



        # annotate MS2 spectra
        if annotation and d.params.msms_library is not None:
            annotate_rois(d)

        if params.plot_bpc:
            d.plot_bpc(label_name=True, output=os.path.join(params.bpc_dir, d.file_name + "_bpc.png"))

        # output single file to a txt file
        if d.params.output_single_file:
            d.output_single_file()

        return d

    except Exception as e:
        print("Error: " + str(e))
        return None

