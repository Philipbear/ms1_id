from ms1id_imaging_workflow import ms1id_imaging_single_workflow
import os


def ms1id_single_file(file_path, msms_library_path, n_processes=None,
                      mass_detect_int_tol=None, max_mz=None,
                      mz_bin_size=0.01,
                      min_spec_overlap_ratio=0.5, min_correlation=0.8, min_cluster_size=6,
                      ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                      ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05):

    ms1id_imaging_single_workflow(file_path=file_path,
                                  msms_library_path=msms_library_path,
                                  n_processes=n_processes,
                                  mass_detect_int_tol=mass_detect_int_tol, max_mz=max_mz,
                                  mz_bin_size=mz_bin_size,
                                  min_spec_overlap_ratio=min_spec_overlap_ratio, min_correlation=min_correlation,
                                  min_cluster_size=min_cluster_size,
                                  ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                                  ms1id_min_matched_peak=ms1id_min_matched_peak,
                                  ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                                  ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)


def ms1id_single_file_batch(file_dir, msms_library_path, n_processes=None,
                            mass_detect_int_tol=None, max_mz=None,
                            mz_bin_size=0.01,
                            min_spec_overlap_ratio=0.5, min_correlation=0.8, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                            ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05):
    files = [f for f in os.listdir(file_dir) if f.endswith('.imzML')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_single_file(file, msms_library_path, n_processes=n_processes,
                          mass_detect_int_tol=mass_detect_int_tol, max_mz=max_mz,
                          mz_bin_size=mz_bin_size,
                          min_spec_overlap_ratio=min_spec_overlap_ratio, min_correlation=min_correlation,
                          min_cluster_size=min_cluster_size,
                          ms1id_mz_tol=ms1id_mz_tol, ms1id_score_cutoff=ms1id_score_cutoff,
                          ms1id_min_matched_peak=ms1id_min_matched_peak,
                          ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                          ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)

    return


if __name__ == '__main__':

    # ############################
    # # Batch workflow
    file_dir = '../../imaging/MTBLS313'
    ms1id_single_file_batch(file_dir=file_dir,
                            msms_library_path='../../data/gnps.pkl',
                            mass_detect_int_tol=None, max_mz=None,
                            mz_bin_size=0.01,
                            min_spec_overlap_ratio=0.8, min_correlation=0.9, min_cluster_size=5,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                            ms1id_min_prec_int_in_ms1=1000, ms1id_max_prec_rel_int_in_other_ms2=0.05)
