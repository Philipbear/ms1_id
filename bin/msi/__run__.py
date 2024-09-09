"""
actual experiment code goes here
"""

from main_msi import ms1id_single_file_batch


##########################################
# 1. MTBLS313, brain
def exp_1():
    ms1id_single_file_batch(file_dir='../../imaging/MTBLS313',
                            library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                            n_processes=12,
                            mass_detect_int_tol=None,
                            noise_detection='moving_average',
                            sn_factor=5.0, centroided=True,
                            mz_bin_size=0.01,
                            min_overlap=10, min_correlation=0.85,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_min_spec_usage=0.10, max_prec_rel_int_in_other_ms2=0.05)


##########################################
# 2. whole mouse body data
def exp_2():
    ms1id_single_file_batch(file_dir='../../imaging/mouse_body',
                            library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                            n_processes=48,
                            mass_detect_int_tol=None,
                            noise_detection='moving_average',
                            sn_factor=10.0, centroided=False,
                            mz_bin_size=0.01,
                            min_overlap=10, min_correlation=0.85,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_min_spec_usage=0.10, max_prec_rel_int_in_other_ms2=0.05)


##########################################
# 3. mouse kidney data
def exp_3():
    ms1id_single_file_batch(file_dir='../../imaging/mouse_kidney',
                            library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                            n_processes=48,
                            mass_detect_int_tol=None,
                            noise_detection='moving_average',
                            sn_factor=10.0, centroided=False,
                            mz_bin_size=0.01,
                            min_overlap=10, min_correlation=0.85,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_min_spec_usage=0.10, max_prec_rel_int_in_other_ms2=0.05)


##########################################
# 4. mouse liver data
def exp_4():
    ms1id_single_file_batch(file_dir='../../imaging/mouse_liver',
                            library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                            n_processes=48,
                            mass_detect_int_tol=None,
                            noise_detection='moving_average',
                            sn_factor=10.0, centroided=False,
                            mz_bin_size=0.01,
                            min_overlap=10, min_correlation=0.85,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_min_spec_usage=0.10, max_prec_rel_int_in_other_ms2=0.05)


if __name__ == '__main__':

    exp_1()

    # exp_2()
    #
    # exp_3()
    #
    # exp_4()
