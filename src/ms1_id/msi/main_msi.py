import os

from ms1_id.msi.ms1id_msi_workflow import ms1id_imaging_workflow


def ms1id_single_file_batch(
        file_dir, library_path, n_processes=None,
        sn_factor=3.0,
        mz_ppm_tol=10.0,
        min_spatial_chaos=0.5,
        min_overlap=30, min_correlation=0.85,
        library_search_mztol=0.01,
        score_cutoff=0.7,
        min_matched_peak=3,
        min_spec_usage=0.10
):
    files = [f for f in os.listdir(file_dir) if f.lower().endswith('.imzml') and not f.startswith('.')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_imaging_workflow(
            file, library_path, n_processes=n_processes,
            sn_factor=sn_factor,
            mz_ppm_tol=mz_ppm_tol,
            min_spatial_chaos=min_spatial_chaos,
            min_overlap=min_overlap, min_correlation=min_correlation,
            library_search_mztol=library_search_mztol,
            score_cutoff=score_cutoff,
            min_matched_peak=min_matched_peak,
            min_spec_usage=min_spec_usage
        )

    return
