"""
actual experiment code goes here
"""

from main_msi import ms1id_single_file_batch


##########################################
# 1. MTBLS313, brain
def exp_1(ms2db):
    ms1id_single_file_batch(file_dir='../../imaging/MTBLS313',
                            library_path=f'../../data/{ms2db}.pkl',
                            n_processes=12,
                            mass_detect_int_tol=None, noise_detection='moving_average',
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.7,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
# 2. whole mouse body data
def exp_2(ms2db):
    ms1id_single_file_batch(file_dir='../../imaging/mouse_body',
                            library_path=f'../../data/{ms2db}.pkl',
                            n_processes=48,
                            mass_detect_int_tol=None, noise_detection='moving_average',
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.7,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
# 3. mouse kidney data
def exp_3(ms2db):
    ms1id_single_file_batch(file_dir='../../imaging/mouse_kidney',
                            library_path=f'../../data/{ms2db}.pkl',
                            n_processes=10,
                            mass_detect_int_tol=None, noise_detection='moving_average',
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.7,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
# 4. mouse liver data
def exp_4(ms2db):
    ms1id_single_file_batch(file_dir='../../imaging/mouse_liver',
                            library_path=f'../../data/{ms2db}.pkl',
                            n_processes=10,
                            mass_detect_int_tol=None, noise_detection='moving_average',
                            mz_bin_size=0.005,
                            min_overlap=10, min_correlation=0.7,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
if __name__ == '__main__':
    exp_1('gnps')

    # exp_2('gnps')

    # exp_3('gnps')

    # exp_4('gnps')
