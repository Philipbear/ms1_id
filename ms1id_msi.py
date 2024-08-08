from bin.msi.main_msi import ms1id_single_file_batch

if __name__ == '__main__':

    file_dir = '../../imaging/MTBLS313'  # file directory containing imzML & ibd files
    msms_library_path = '../../data/gnps.pkl'

    ms1id_single_file_batch(file_dir=file_dir,
                            msms_library_path=msms_library_path,
                            n_processes=None,
                            mass_detect_int_tol=None, max_mz=None, mz_bin_size=0.01,
                            min_spec_overlap_ratio=0.5, min_correlation=0.9, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_min_prec_int_in_ms1=0)
