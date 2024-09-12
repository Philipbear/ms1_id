"""
actual experiment code goes here
"""

from main_lcms import ms1id_single_file_batch, ms1id_batch_mode


def std():
    # for k in ['_k0', '_k3', '_k5', '_k8', '_k9', '_k10', '_k13', '_k15', '_k18']:
    #     ms1id_single_file_batch('../../data/std_mix/data_1',
    #                             f'../../data/std_mix/std{k}.pkl',
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
    #                             out_dir=f'../../bin/lcms/analysis/std/output{k}')

    ms1id_single_file_batch('../../data/std_mix/data_1',
                            ['../../data/std_mix/std_k0.pkl',
                             '../../data/std_mix/std_k10.pkl'],
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
                            out_dir='../../bin/lcms/analysis/std/output_all')


##########################################
# 1. NIST pool sample, full scan, 0eV, 10 eV, 20 eV
def exp_1():
    mass_detection_int_tol = 1.5e5
    # Full scan, MS1 annotation
    for ev in [0, 10, 20]:
        ms1id_batch_mode(project_path=f'../../data/nist/fullscan_{ev}ev',
                         ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                         sample_dir='data', parallel=True,
                         ms1_id=True, ms2_id=False,
                         cpu_ratio=0.90,
                         run_rt_correction=True, run_normalization=True,
                         ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                         align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                         peak_cor_rt_tol=0.025,
                         min_ppc=0.8, roi_min_length=4,
                         library_search_mztol=0.05,
                         ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                         ms1id_min_spec_usage=0.20,
                         ms1id_max_prec_rel_int_in_other_ms2=0.01,
                         ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


##########################################
# 2. NIST pool sample, DDA
def exp_2():
    mass_detection_int_tol = 1.5e5
    # DDA, MS/MS annotation
    ms1id_batch_mode(project_path='../../data/nist/dda_ms2',
                     ms2id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=False, ms2_id=True,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    # DDA, MS1 annotation
    ms1id_batch_mode(project_path='../../data/nist/dda_ms1',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


##########################################
# 3. iHMP pool sample, full scan
def exp_3():
    mass_detection_int_tol = 5e5
    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_pos',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c18_neg',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_neg',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c8_pos',
                     ms1id_library_path=['../../data/gnps.pkl', '../../data/gnps_k10.pkl'],
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.90,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=4,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_min_spec_usage=0.20,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


if __name__ == '__main__':

    # std()

    exp_1()

    exp_2()

    # exp_3()
