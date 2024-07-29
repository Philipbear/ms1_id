from ms1id_imaging_workflow import ms1id_imaging_single_workflow
import os
import multiprocessing
from functools import partial


def ms1id_single_file(file_path, msms_library_path,
                      mass_detect_int_tol=None, mz_bin_size=0.01,
                      min_spec_overlap_ratio=0.5, min_correlation=0.8, min_cluster_size=6,
                      ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                      ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05):

    ms1id_imaging_single_workflow(file_path=file_path,
                                  msms_library_path=msms_library_path,
                                  mass_detect_int_tol=mass_detect_int_tol, mz_bin_size=mz_bin_size,
                                  min_spec_overlap_ratio=min_spec_overlap_ratio, min_correlation=min_correlation,
                                  min_cluster_size=min_cluster_size,
                                  ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                                  ms1id_min_matched_peak=ms1id_min_matched_peak,
                                  ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                                  ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)


def ms1id_single_file_batch(file_dir, msms_library_path, parallel=True, num_processes=None,
                            mass_detect_int_tol=None, mz_bin_size=0.01,
                            min_spec_overlap_ratio=0.5, min_correlation=0.8, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                            ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05):
    files = [f for f in os.listdir(file_dir) if f.endswith('.imzML')]
    files = [os.path.join(file_dir, f) for f in files]

    if parallel:
        # If num_processes is not specified, use the number of CPU cores
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Create a partial function with the library_path argument
        process_file = partial(ms1id_single_file, msms_library_path=msms_library_path,
                               mass_detect_int_tol=mass_detect_int_tol, mz_bin_size=mz_bin_size,
                               min_spec_overlap_ratio=min_spec_overlap_ratio, min_correlation=min_correlation,
                               min_cluster_size=min_cluster_size,
                               ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                               ms1id_min_matched_peak=ms1id_min_matched_peak,
                               ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                               ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)

        # Create a pool of workers
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_file, files)
    else:
        for file in files:
            ms1id_single_file(file, msms_library_path,
                              mass_detect_int_tol=mass_detect_int_tol, mz_bin_size=mz_bin_size,
                              min_spec_overlap_ratio=min_spec_overlap_ratio, min_correlation=min_correlation,
                              min_cluster_size=min_cluster_size,
                              ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                              ms1id_min_matched_peak=ms1id_min_matched_peak,
                              ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                              ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)

    return


if __name__ == '__main__':
    ############################
    # Single
    file_path = '../../imaging/MTBLS313/Brain02_Bregma-3-88.imzML'
    msms_library_path = '../../data/gnps_nist20.pkl'
    ms1id_single_file(file_path=file_path, msms_library_path=msms_library_path,
                      mass_detect_int_tol=None, mz_bin_size=0.01,
                      min_spec_overlap_ratio=0.5, min_correlation=0.9, min_cluster_size=6,
                      ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                      ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05)

    # ############################
    # # Batch workflow
    # file_dir = '../../imaging/MTBLS313'
    # ms1id_single_file_batch(file_dir=file_dir,
    #                         msms_library_path='../../data/gnps_nist20.pkl',
    #                         parallel=False, num_processes=None,
    #                         mass_detect_int_tol=None, mz_bin_size=0.01,
    #                         min_spec_overlap_ratio=0.5, min_correlation=0.9, min_cluster_size=6,
    #                         ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
    #                         ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05)
