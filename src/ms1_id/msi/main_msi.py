import os

from ms1_id.msi.ms1id_msi_workflow import ms1id_imaging_workflow


def ms1id_single_file_batch(
        file_dir, library_path, n_processes=None,
        sn_factor=5.0,
        mz_bin_size=0.01,
        min_overlap=10, min_correlation=0.85, max_cor_depth=1,
        library_search_mztol=0.01,
        ms1id_score_cutoff=0.7,
        ms1id_min_matched_peak=4,
        ms1id_min_spec_usage=0.10,
        max_prec_rel_int_in_other_ms2=0.05
):
    files = [f for f in os.listdir(file_dir) if f.lower().endswith('.imzml') and not f.startswith('.')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_imaging_workflow(
            file, library_path, n_processes=n_processes,
            sn_factor=sn_factor,
            mz_bin_size=mz_bin_size,
            min_overlap=min_overlap, min_correlation=min_correlation,
            max_cor_depth=max_cor_depth,
            library_search_mztol=library_search_mztol, ms1id_score_cutoff=ms1id_score_cutoff,
            ms1id_min_spec_usage=ms1id_min_spec_usage,
            ms1id_min_matched_peak=ms1id_min_matched_peak,
            max_prec_rel_int_in_other_ms2=max_prec_rel_int_in_other_ms2
        )

    return
