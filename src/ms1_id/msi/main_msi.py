import os

from ms1_id.msi.ms1id_msi_workflow import ms1id_imaging_workflow


def ms1id_single_file_batch(
        file_dir, library_path,
        polarity=None,
        n_processes=None,
        sn_factor=3.0,
        mz_ppm_tol=10.0,
        min_feature_spatial_chaos=0.10,
        min_pixel_overlap=50, min_correlation=0.85,
        library_search_mztol=0.01,
        score_cutoff=0.7,
        min_matched_peak=3,
        min_spec_usage=0.10
):
    files = [f for f in os.listdir(file_dir) if f.lower().endswith('.imzml') and not f.startswith('.')]
    files = [os.path.join(file_dir, f) for f in files]

    for file in files:
        ms1id_imaging_workflow(
            file, library_path,
            polarity=polarity,
            n_processes=n_processes,
            sn_factor=sn_factor,
            mz_ppm_tol=mz_ppm_tol,
            min_feature_spatial_chaos=min_feature_spatial_chaos,
            min_pixel_overlap=min_pixel_overlap, min_correlation=min_correlation,
            library_search_mztol=library_search_mztol,
            score_cutoff=score_cutoff,
            min_matched_peak=min_matched_peak,
            min_spec_usage=min_spec_usage
        )

    return
