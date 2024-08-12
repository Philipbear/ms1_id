"""
actual experiment code goes here
"""

from main_msi import ms1id_single_file_batch


##########################################
# 1. MTBLS313, brain
def exp_1(ms2db='gnps'):
    ms1id_single_file_batch(file_dir='../../imaging/MTBLS313',
                            msms_library_path=f'../../data/{ms2db}.pkl',
                            n_processes=8,
                            mass_detect_int_tol=None, max_mz=None,
                            mz_bin_size=0.01,
                            min_overlap=5, min_correlation=0.9, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
# 2. whole mouse body data
def exp_2(ms2db='gnps'):
    ms1id_single_file_batch(file_dir='../../imaging/mouse_body',
                            msms_library_path=f'../../data/{ms2db}_k10.pkl',
                            n_processes=48,
                            mass_detect_int_tol=5e5, max_mz=None,
                            mz_bin_size=0.01,
                            min_overlap=5, min_correlation=0.9, min_cluster_size=6,
                            ms1id_mz_tol=0.01, ms1id_score_cutoff=0.6, ms1id_min_matched_peak=3)


##########################################
if __name__ == '__main__':
    # exp_1('gnps')
    exp_2('gnps')
