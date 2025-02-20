"""
create a workflow for MS1_ID using masscube backend.
for multiple files.
"""

import multiprocessing
import os

from masscube.alignment import feature_alignment, gap_filling
from masscube.feature_grouping import annotate_isotope
from masscube.feature_table_utils import convert_features_to_df
from masscube.normalization import sample_normalization
from masscube.params import Params
# from masscube.raw_data_utils import MSData
from ms1_id.lcms.masscube_raw_data_utils import MSData

from ms1_id.lcms.annotate_adduct import annotate_adduct
from ms1_id.lcms.annotate_ms2 import feature_annotation
from ms1_id.lcms.calculate_ppc import calc_all_ppc
from ms1_id.lcms.export import write_feature_table
from ms1_id.lcms.group_ppc import generate_pseudo_ms2, retrieve_pseudo_ms2_spectra
from ms1_id.lcms.reverse_matching import ms1_id_annotation
from ms1_id.lcms.utils import find_ms_info

# default parameters
orbitrap_mass_detect_int_tol = 20000
tof_mass_detect_int_tol = 500


def main_workflow(project_path=None, ms1id_library_path=None, ms2id_library_path=None,
                  sample_dir='data', parallel=True,
                  ms1_id=True, ms2_id=False,
                  cpu_ratio=0.9,
                  run_rt_correction=True, run_normalization=False,
                  ms1_tol=0.01, ms2_tol=0.015, mass_detect_int_tol=None,
                  align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                  peak_cor_rt_tol=0.015,
                  min_ppc=0.6, roi_min_length=3,
                  library_search_mztol=0.05,
                  ms1id_score_cutoff=0.8, ms1id_min_matched_peak=6, ms1id_min_spec_usage=0.05,
                  ms1id_max_prec_rel_int_in_other_ms2=0.01,
                  ms2id_score_cutoff=0.8, ms2id_min_matched_peak=6):
    """
    Main workflow for MS1_ID.
    :param project_path: path to the project directory
    :param ms1id_library_path: path to the MS1 ID library
    :param ms2id_library_path: path to the MS2 ID library
    :param parallel: whether to run in parallel
    :param ms1_id: whether to perform MS1 ID
    :param ms2_id: whether to perform MS2 ID
    :param cpu_ratio: CPU ratio for multiprocessing
    :param sample_dir: directory where the raw files are stored
    :param run_rt_correction: whether to perform RT correction
    :param run_normalization: whether to perform normalization
    :param ms1_tol: m/z tolerance for MS1
    :param ms2_tol: m/z tolerance for MS2
    :param mass_detect_int_tol: intensity tolerance for mass detection
    :param align_mz_tol: alignment m/z tolerance
    :param align_rt_tol: alignment RT tolerance
    :param alignment_drop_by_fill_pct_ratio: alignment drop by fill percentage ratio
    :param peak_cor_rt_tol: peak correlation RT tolerance
    :param min_ppc: minimum peak-peak correlation threshold
    :param roi_min_length: minimum ROI length for feature grouping
    :param ms1id_score_cutoff: ms1 ID score cutoff
    :param ms1id_min_matched_peak: ms1 ID min matched peak
    :param ms1id_max_prec_rel_int_in_other_ms2: ms1 ID, max precursor relative intensity in other MS2
    :param ms2id_score_cutoff: ms2 ID score cutoff
    :param ms2id_min_matched_peak: ms2 ID min matched peak
    :return:
    """

    # init a new config object
    config = init_config(project_path, ms1id_library_path, ms2id_library_path,
                         sample_dir,
                         ms1_id=ms1_id, ms2_id=ms2_id,
                         run_rt_correction=run_rt_correction, run_normalization=run_normalization,
                         mz_tol_ms1=ms1_tol, mz_tol_ms2=ms2_tol, mass_detect_int_tol=mass_detect_int_tol,
                         align_mz_tol=align_mz_tol, align_rt_tol=align_rt_tol,
                         alignment_drop_by_fill_pct_ratio=alignment_drop_by_fill_pct_ratio,
                         peak_cor_rt_tol=peak_cor_rt_tol,
                         min_ppc=min_ppc, roi_min_length=roi_min_length,
                         library_search_mztol=library_search_mztol,
                         ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                         ms1id_min_spec_usage=ms1id_min_spec_usage,
                         ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                         ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak)

    raw_file_names = os.listdir(config.sample_dir)
    raw_file_names = [f for f in raw_file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    raw_file_names = [f for f in raw_file_names if not f.startswith(".")]  # for Mac OS
    total_file_num = len(raw_file_names)

    # skip the files that have been processed
    txt_files = os.listdir(config.single_file_dir)
    txt_files = [f.split(".")[0] for f in txt_files if f.lower().endswith(".txt") and not f.startswith(".")]
    raw_file_names = [f for f in raw_file_names if f.split(".")[0] not in txt_files]
    raw_file_names = [os.path.join(config.sample_dir, f) for f in raw_file_names]

    print("\t{} raw file are found, {} files to be processed.".format(total_file_num, len(raw_file_names)))

    if len(raw_file_names) > 0:
        # process files by multiprocessing
        print("Processing individual files for feature detection, evaluation, and grouping...")

        if parallel:
            workers = min(int(multiprocessing.cpu_count() * cpu_ratio), len(raw_file_names))
            print("\t {} CPU cores in total, {} cores used.".format(multiprocessing.cpu_count(), workers))

            print(f"Processing all {len(raw_file_names)} files")
            p = multiprocessing.Pool(workers)
            p.starmap(feature_detection, [(f, config) for f in raw_file_names])
            p.close()
            p.join()
        else:
            for i, file_name in enumerate(raw_file_names):
                feature_detection(file_name, params=config)

    pseudo_ms2_spectra = []
    if ms1_id:
        # get all pre-saved pseudo ms2 spectra
        pseudo_ms2_spectra = retrieve_pseudo_ms2_spectra(config)

    # feature alignment
    print("Aligning features...")
    features = feature_alignment(config.single_file_dir, config,
                                 drop_by_fill_pct_ratio=alignment_drop_by_fill_pct_ratio)

    # gap filling
    print("Filling gaps...")
    features = gap_filling(features, config)

    # annotation (using MS2 library)
    if ms2_id and config.ms2id_library_path is not None:
        print("Annotating MS2...")
        features = feature_annotation(features, config)

    feature_table = convert_features_to_df(features, config.sample_names)

    # normalization
    if config.run_normalization:
        print("Running normalization...")
        feature_table = sample_normalization(feature_table, config.individual_sample_groups,
                                             config.normalization_method)

    # output feature table
    output_path = os.path.join(config.project_dir, "aligned_feature_table.tsv")
    write_feature_table(feature_table, pseudo_ms2_spectra, config, output_path)

    print("The workflow is completed.")


def init_config(path=None,
                ms1id_library_path=None, ms2id_library_path=None,
                sample_dir='data',
                ms1_id=True, ms2_id=False,
                run_rt_correction=True, run_normalization=False,
                mz_tol_ms1=0.01, mz_tol_ms2=0.015, mass_detect_int_tol=None,
                align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                peak_cor_rt_tol=0.015,
                min_ppc=0.6, roi_min_length=3,
                library_search_mztol=0.05,
                ms1id_score_cutoff=0.8, ms1id_min_matched_peak=6, ms1id_min_spec_usage=0.05,
                ms1id_max_prec_rel_int_in_other_ms2=0.01,
                ms2id_score_cutoff=0.8, ms2id_min_matched_peak=6
                ):
    # init
    config = Params()

    config.ms1_id = ms1_id
    config.ms2_id = ms2_id

    config.peak_cor_rt_tol = peak_cor_rt_tol
    config.min_ppc = min_ppc
    config.roi_min_length = roi_min_length
    config.ms1id_score_cutoff = ms1id_score_cutoff
    config.ms1id_min_matched_peak = ms1id_min_matched_peak
    config.ms1id_min_spec_usage = ms1id_min_spec_usage
    config.ms1id_max_prec_rel_int_in_other_ms2 = ms1id_max_prec_rel_int_in_other_ms2
    config.ms2id_score_cutoff = ms2id_score_cutoff
    config.ms2id_min_matched_peak = ms2id_min_matched_peak

    # obtain the working directory
    config.project_dir = path
    print("Project directory: " + config.project_dir)

    config.sample_dir = os.path.join(config.project_dir, sample_dir)
    print("Sample directory: " + config.sample_dir)

    # check if the required files are prepared
    if not os.path.exists(config.sample_dir) or len(os.listdir(config.sample_dir)) == 0:
        raise ValueError("No raw MS data is found in the sample directory.")

    config.single_file_dir = os.path.join(config.project_dir, "single_files")
    os.makedirs(config.single_file_dir, exist_ok=True)

    # determine the type of MS and ion mode
    file_names = os.listdir(config.sample_dir)
    file_names = [f for f in file_names if f.lower().endswith(".mzml") or f.lower().endswith(".mzxml")]
    file_names = [f for f in file_names if not f.startswith(".")]  # for Mac OS
    file_name = os.path.join(config.sample_dir, file_names[0])
    ms_type, ion_mode, _ = find_ms_info(file_name)

    if mass_detect_int_tol is not None:
        config.int_tol = mass_detect_int_tol
    else:
        if ms_type == "orbitrap":
            config.int_tol = orbitrap_mass_detect_int_tol
        elif ms_type == "tof":
            config.int_tol = tof_mass_detect_int_tol
        else:
            config.int_tol = 20000

    ##########################
    # The project
    # config.project_dir = None  # Project directory, character string
    config.sample_names = [f.split(".")[0] for f in os.listdir(config.sample_dir)
                           if not f.startswith(".")]
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
    # config.int_tol = 30000  # Intensity tolerance, default is 30000 for Orbitrap and 1000 for other instruments
    config.roi_gap = 30  # Gap within a feature, default is 30 (i.e. 30 consecutive scans without signal), integer
    config.ppr = 0.7  # Peak-peak correlation threshold for feature grouping, default is 0.7

    # Parameters for feature alignment
    config.align_mz_tol = align_mz_tol  # m/z tolerance for MS1, default is 0.01
    config.align_rt_tol = align_rt_tol  # RT tolerance, default is 0.2
    config.alignment_drop_by_fill_pct_ratio = alignment_drop_by_fill_pct_ratio  # Drop by fill percentage ratio, default is 0.1
    config.run_rt_correction = run_rt_correction  # Whether to perform RT correction, default is True
    config.min_scan_num_for_alignment = 10  # Minimum scan number a feature to be aligned, default is 6

    # Parameters for feature annotation
    config.ms1id_library_path = ms1id_library_path
    config.library_search_mztol = library_search_mztol  # m/z tolerance for MS1 ID, default is 0.05
    config.ms2id_library_path = ms2id_library_path
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


def feature_detection(file_name, params=None,
                      cal_g_score=False, cal_a_score=False,
                      anno_isotope=True, anno_adduct=True, anno_in_source_fragment=False,
                      ms2_library_path=None, cut_roi=True):
    """
    Untargeted feature detection from a single file (.mzML or .mzXML).

    Returns: An MSData object containing the processed data.
    """
    print("=========================================")
    print("Processing file: " + file_name)

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
        params.ms2id_library_path = ms2_library_path

    print(f'Reading raw data from {file_name}...')
    # read raw data
    d.read_raw_data(file_name, params)

    print(f"Detecting features from {file_name}...")
    # detect region of interests (ROIs)
    d.find_rois()

    # cut ROIs
    if cut_roi:
        d.cut_rois()

    print(f"Summarizing features from {file_name}...")
    # label short ROIs, find the best MS2, and sort ROIs by m/z
    d.summarize_roi(cal_g_score=cal_g_score, cal_a_score=cal_a_score)

    if anno_isotope:
        print(f"Annotating isotopes from {file_name}...")
        annotate_isotope(d, mz_tol=0.015, rt_tol=params.peak_cor_rt_tol)
    # if anno_in_source_fragment:
    #     annotate_in_source_fragment(d)
    if anno_adduct:
        print(f"Annotating adducts from {file_name}...")
        annotate_adduct(d, mz_tol=0.01, rt_tol=params.peak_cor_rt_tol)

    if params.ms1_id:
        print(f"Calculating peak-peak correlations for {file_name}...")
        # calc peak-peak correlations for feature groups and output
        ppc_matrix = calc_all_ppc(d, rt_tol=params.peak_cor_rt_tol,
                                  roi_min_length=params.roi_min_length,
                                  save=False)

        print(f"Generating and saving pseudo MS2 spectra for {file_name}...")
        # generate pseudo ms2 spec, for ms1_id
        pseudo_ms2_spectra = generate_pseudo_ms2(d, ppc_matrix,
                                                 mz_tol=params.mz_tol_ms1,
                                                 min_ppc=params.min_ppc,
                                                 min_cluster_size=params.ms1id_min_matched_peak,
                                                 roi_min_length=params.roi_min_length,
                                                 save_dir=params.single_file_dir)
        del ppc_matrix

        print(f"Performing MS1 ID annotation for {file_name}...")
        # perform rev cos search
        ms1_id_annotation(pseudo_ms2_spectra, params.ms1id_library_path,
                          mz_tol=params.library_search_mztol,
                          ion_mode=params.ion_mode,
                          max_prec_rel_int_in_other_ms2=params.ms1id_max_prec_rel_int_in_other_ms2,
                          score_cutoff=params.ms1id_score_cutoff,
                          min_matched_peak=params.ms1id_min_matched_peak,
                          min_spec_usage=params.ms1id_min_spec_usage,
                          save=True,
                          save_path=os.path.join(params.single_file_dir,
                                                 os.path.basename(file_name).split(".")[
                                                     0] + "_pseudoMS2_annotated.pkl"))
        del pseudo_ms2_spectra

    # output single file to a txt file
    d.output_single_file()

    return d
