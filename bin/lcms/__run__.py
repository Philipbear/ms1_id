"""
actual experiment code goes here
"""

from main_lcms import ms1id_single_file_batch, ms1id_batch_mode


def std():
    # for k in ['_k0', '_k3', '_k5', '_k8', '_k9', '_k10', '_k13', '_k15', '_k18']:
    #     ms1id_single_file_batch('/Users/shipei/Documents/projects/ms1_id/data/std_mix/data_1',
    #                             f'/Users/shipei/Documents/projects/ms1_id/data/std_mix/std{k}.pkl',
    #                             parallel=True, num_processes=None,
    #                             ms1_id=True, ms2_id=False,
    #                             ms1_tol=0.01, ms2_tol=0.015,
    #                             mass_detect_int_tol=30000,
    #                             peak_cor_rt_tol=0.025,
    #                             min_ppc=0.8, roi_min_length=4,
    #                             library_search_mztol=0.05,
    #                             ms1id_score_cutoff=0.01, ms1id_min_matched_peak=2,
    #                             ms1id_max_prec_rel_int_in_other_ms2=0.01,
    #                             ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
    #                             out_dir=f'/Users/shipei/Documents/projects/ms1_id/bin/lcms/analysis/std/output{k}')

    ms1id_single_file_batch('/Users/shipei/Documents/projects/ms1_id/data/std_mix/data_1',
                            ['/Users/shipei/Documents/projects/ms1_id/data/std_mix/std_k0.pkl',
                             '/Users/shipei/Documents/projects/ms1_id/data/std_mix/std_k10.pkl'],
                            parallel=True, num_processes=None,
                            ms1_id=True, ms2_id=False,
                            ms1_tol=0.01, ms2_tol=0.015,
                            mass_detect_int_tol=30000,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.01, ms1id_min_matched_peak=2,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                            out_dir='/Users/shipei/Documents/projects/ms1_id/bin/lcms/analysis/std/output_all')


##########################################
# 1. NIST pool sample, full scan, 0eV, 10 eV, 20 eV
def exp_1():
    mass_detection_int_tol = 2e5
    # ms1id_single_file_batch(data_dir='../../data/nist/data_1',
    #                         ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
    #                         parallel=True, num_processes=12,
    #                         ms1_id=True, ms2_id=False,
    #                         ms1_tol=0.01, ms2_tol=0.02,
    #                         mass_detect_int_tol=mass_detection_int_tol,
    #                         peak_cor_rt_tol=0.025,
    #                         min_ppc=0.8, roi_min_length=4,
    #                         library_search_mztol=0.05,
    #                         ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
    #                         ms1id_max_prec_rel_int_in_other_ms2=0.01,
    #                         ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
    #                         out_dir='../../data/nist/data_1/output_gnps')

    ms1id_single_file_batch(data_dir='../../data/nist/data_1',
                            ms1id_library_path=['../../data/gnps.pkl'],
                            parallel=True, num_processes=12,
                            ms1_id=True, ms2_id=False,
                            ms1_tol=0.01, ms2_tol=0.02,
                            mass_detect_int_tol=mass_detection_int_tol,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
                            out_dir='../../data/nist/data_1/output_gnps_k0')


##########################################
# 2. NIST pool sample, DDA
def exp_2():
    mass_detection_int_tol = 2e5
    ms1id_single_file_batch(data_dir='../../data/nist/data_2',
                            ms2id_library_path='../../data/gnps.pkl',
                            parallel=True, num_processes=6,
                            ms1_id=False, ms2_id=True,
                            ms1_tol=0.01, ms2_tol=0.02,
                            mass_detect_int_tol=mass_detection_int_tol,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
                            out_dir='../../data/nist/data_2/output_gnps')


##########################################
# 3. iHMP pool sample, full scan
def exp_3():
    mass_detection_int_tol = 5e5
    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_pos',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.9,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c18_neg',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.9,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_neg',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.9,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c8_pos',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.9,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


if __name__ == '__main__':

    # std()

    # exp_1()

    # exp_2()

    exp_3()
