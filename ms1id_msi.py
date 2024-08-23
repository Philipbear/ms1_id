from bin.msi.main_msi import ms1id_single_file_batch

if __name__ == '__main__':

    file_dir = '../../imaging/MTBLS313'  # file directory containing imzML & ibd files
    library_path1 = '../../data/gnps_k10.pkl'
    library_path2 = '../../data/gnps.pkl'

    ms1id_single_file_batch(file_dir=file_dir,
                            library_path=[library_path1, library_path2],
                            n_processes=None,
                            mass_detect_int_tol=None, noise_detection='moving_average',
                            sn_factor=5.0, centroided=True,
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.80, max_cor_depth=1,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)
