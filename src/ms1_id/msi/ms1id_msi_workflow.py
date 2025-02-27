import os
import multiprocessing as mp

from ms1_id.msi.calculate_mz_cor_parallel import calc_all_mz_correlations
from ms1_id.msi.export_msi import write_ms1_id_results
from ms1_id.msi.group_mz_cor_parallel import generate_pseudo_ms2
from ms1_id.msi.process_msi_data import process_ms_imaging_data
from ms1_id.msi.reverse_matching_parallel import validate_library_path, ms1_id_annotation


def ms1id_imaging_workflow(file_path, library_path,
                           polarity=None,
                           n_processes=None,
                           sn_factor=5.0,
                           mz_ppm_tol=5.0,
                           min_feature_spatial_chaos=0.01,
                           min_pixel_overlap=50, min_correlation=0.85,
                           library_search_mztol=0.05,
                           score_cutoff=0.7, min_matched_peak=4,
                           min_spec_usage=0.10):
    file_dir = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # validate library_path
    library_path = validate_library_path(library_path)

    n_processes = n_processes or mp.cpu_count()

    # make a result folder of file_name
    result_folder = os.path.join(file_dir, file_name)
    os.makedirs(result_folder, exist_ok=True)

    print(f"Processing {file_name}")
    feature_ls, intensity_matrix, ion_mode = process_ms_imaging_data(
        file_path,
        os.path.splitext(file_path)[0] + '.ibd',
        polarity=polarity,
        mz_ppm_tol=mz_ppm_tol,
        sn_factor=sn_factor,
        min_feature_pixel_count=min_pixel_overlap,
        min_feature_spatial_chaos=min_feature_spatial_chaos,
        n_processes=n_processes,
        save_dir=result_folder
    )

    print(f"Calculating ion image correlations for {file_name}")
    cor_matrix = calc_all_mz_correlations(intensity_matrix,
                                          min_pixel_overlap=min_pixel_overlap,
                                          min_cor=min_correlation,
                                          n_processes=n_processes,
                                          save_dir=result_folder)

    print(f"Generating pseudo MS2 spectra for {file_name}")
    pseudo_ms2 = generate_pseudo_ms2(feature_ls, intensity_matrix, cor_matrix,
                                     n_processes=n_processes,
                                     min_cluster_size=min_matched_peak + 1,
                                     min_cor=min_correlation,
                                     save_dir=result_folder)

    print(f"Annotating pseudo MS2 spectra for {file_name}")
    pseudo_ms2 = ms1_id_annotation(pseudo_ms2, library_path, n_processes=n_processes,
                                   library_search_mz_tol=library_search_mztol,
                                   ms1_ppm_tol=mz_ppm_tol,
                                   ion_mode=ion_mode,
                                   score_cutoff=score_cutoff,
                                   min_matched_peak=min_matched_peak,
                                   min_spec_usage=min_spec_usage,
                                   save_dir=result_folder)

    print(f"Writing results for {file_name}")
    write_ms1_id_results(feature_ls, pseudo_ms2, save=True, save_dir=result_folder)

    return
