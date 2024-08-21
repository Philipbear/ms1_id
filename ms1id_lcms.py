from bin.lcms.main_lcms import ms1id_batch_mode

if __name__ == '__main__':

    project_path = '../../data/MSV000087562/C18_neg_iHMPpool'
    library_path1 = '../../data/gnps_k10.pkl'
    library_path2 = '../../data/gnps.pkl'

    # mzML/mzXML files should be in '{project_path}/{sample_dir}' directory

    ms1id_batch_mode(project_path=project_path,
                     ms1id_library_path=[library_path1, library_path2],
                     ms2id_library_path=library_path2,
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.8,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.02, mass_detect_int_tol=30000,
                     align_mz_tol=0.01, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5,
                     library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4)
