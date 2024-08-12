from ms1id_msi_workflow import ms1id_imaging_single_workflow
import os


def ms1id_single_file_batch(file_dir, msms_library_path, n_processes=None,
                            mass_detect_int_tol=None, max_mz=None,
                            mz_bin_size=0.01,
                            min_overlap=5, min_correlation=0.8, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=4):
    files = [f for f in os.listdir(file_dir) if f.endswith('.imzML')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_imaging_single_workflow(file, msms_library_path, n_processes=n_processes,
                                      mass_detect_int_tol=mass_detect_int_tol, max_mz=max_mz,
                                      mz_bin_size=mz_bin_size,
                                      min_overlap=min_overlap, min_correlation=min_correlation,
                                      min_cluster_size=min_cluster_size,
                                      ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                                      ms1id_min_matched_peak=ms1id_min_matched_peak)

    return

