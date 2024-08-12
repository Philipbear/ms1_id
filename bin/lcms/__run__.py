"""
actual experiment code goes here
"""

from main_lcms import ms1id_single_file_batch, ms1id_batch_mode


def std():
    ms1id_single_file_batch('/Users/shipei/Documents/projects/ms1_id/data/std_mix/data_1',
                            '/Users/shipei/Documents/projects/ms1_id/data/std_mix/std_k10.pkl',
                            parallel=True, num_processes=None,
                            ms1_id=True, ms2_id=False,
                            mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                            mass_detect_int_tol=30000,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            ms1id_score_cutoff=0.01, ms1id_min_matched_peak=2,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                            out_dir='/Users/shipei/Documents/projects/ms1_id/bin/lcms/analysis/std/output_scale_k10')

    ms1id_single_file_batch('/Users/shipei/Documents/projects/ms1_id/data/std_mix/data_1',
                            '/Users/shipei/Documents/projects/ms1_id/data/std_mix/std.pkl',
                            parallel=True, num_processes=None,
                            ms1_id=True, ms2_id=False,
                            mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                            mass_detect_int_tol=30000,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            ms1id_score_cutoff=0.01, ms1id_min_matched_peak=2,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                            out_dir='/Users/shipei/Documents/projects/ms1_id/bin/lcms/analysis/std/output_noscale')


##########################################
# 1. NIST pool sample, full scan, 0eV, 10 eV, 20 eV
def exp_1(ms2db='gnps'):
    mass_detection_int_tol = 200000
    ms1id_single_file_batch(data_dir='../../data/nist/data_1',
                            ms1id_library_path=f'../../data/{ms2db}_k10.pkl',
                            parallel=True, num_processes=6,
                            ms1_id=True, ms2_id=False,
                            mz_tol_ms1=0.01, mz_tol_ms2=0.02,
                            mass_detect_int_tol=mass_detection_int_tol,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=5,
                            ms1id_score_cutoff=0.6, ms1id_min_matched_peak=4,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.6, ms2id_min_matched_peak=4,
                            out_dir=f'../../data/nist/data_1/output_k10')


##########################################
# 2. NIST pool sample, DDA
def exp_2(ms2db='gnps'):
    mass_detection_int_tol = 200000
    ms1id_single_file_batch(data_dir='../../data/nist_samples/data_2',
                            ms1id_library_path=f'../../data/{ms2db}.pkl',
                            parallel=True, num_processes=3,
                            ms1_id=False, ms2_id=True,
                            mz_tol_ms1=0.01, mz_tol_ms2=0.02,
                            mass_detect_int_tol=mass_detection_int_tol,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=5,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
                            out_dir=f'../../data/nist_samples/data_2/output_{ms2db}')


##########################################
# 3. iHMP pool sample, DDA
def exp_3():
    mass_detection_int_tol = 200000
    ms1id_batch_mode(project_path='../../data/MSV000087562/C18_neg_iHMPpool',
                     ms2id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=False, ms2_id=True,
                     cpu_ratio=0.8,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/MSV000087562/HILIC_pos_iHMPpool',
                     ms2id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=False, ms2_id=True,
                     cpu_ratio=0.8,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


##########################################
# 3. iHMP pool sample, full scan
def exp_4():
    mass_detection_int_tol = 500000
    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_pos',
                     ms1id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.6,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c18_neg',
                     ms1id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.6,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/hilic_neg',
                     ms1id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.6,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)

    ms1id_batch_mode(project_path='../../data/PR000639_data/c8_pos',
                     ms1id_library_path='../../data/gnps.pkl',
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.6,
                     run_rt_correction=True, run_normalization=True,
                     mz_tol_ms1=0.01, mz_tol_ms2=0.02, mass_detect_int_tol=mass_detection_int_tol,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)


if __name__ == '__main__':

    std()
    # exp_1('gnps')
    # exp_1('nist20')

    # exp_2('gnps')
    # exp_2('nist20')

    # exp_3()

    # exp_4()
