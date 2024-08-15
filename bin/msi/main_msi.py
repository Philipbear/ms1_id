from ms1id_msi_workflow import ms1id_imaging_single_workflow
import os


def ms1id_single_file_batch(file_dir, library_path, n_processes=None,
                            mass_detect_int_tol=None,
                            noise_detection='moving_average',
                            sn_factor=5.0,
                            centroided=True,
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.80, max_cor_depth=1,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3):
    files = [f for f in os.listdir(file_dir) if f.endswith('.imzML')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_imaging_single_workflow(file, library_path, n_processes=n_processes,
                                      mass_detect_int_tol=mass_detect_int_tol,
                                      noise_detection=noise_detection,
                                      sn_factor=sn_factor,
                                      centroided=centroided,
                                      mz_bin_size=mz_bin_size,
                                      min_overlap=min_overlap, min_correlation=min_correlation,
                                      max_cor_depth=max_cor_depth,
                                      ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                                      ms1id_min_matched_peak=ms1id_min_matched_peak)

    return
