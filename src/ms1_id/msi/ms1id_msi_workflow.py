import os

from ms1_id.msi.calculate_mz_cor_parallel import calc_all_mz_correlations
from ms1_id.msi.export_msi import write_ms1_id_results
from ms1_id.msi.group_mz_cor_parallel import generate_pseudo_ms2
from ms1_id.msi.process_msi_data import process_ms_imaging_data
from ms1_id.msi.reverse_matching_parallel import validate_library_path, ms1_id_annotation


def ms1id_imaging_workflow(file_path, library_path, n_processes=None,
                           mass_detect_int_tol=None, noise_detection='moving_average',
                           sn_factor=5.0, centroided=False,
                           mz_bin_size=0.01,
                           min_overlap=10, min_correlation=0.85, max_cor_depth=1,
                           library_search_mztol=0.05,
                           ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                           ms1id_min_spec_usage=0.10, max_prec_rel_int_in_other_ms2=0.05):
    file_dir = os.path.dirname(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # validate library_path
    library_path = validate_library_path(library_path)

    # make a result folder of file_name
    result_folder = os.path.join(file_dir, file_name)
    os.makedirs(result_folder, exist_ok=True)

    print(f"Processing {file_name}")
    mz_values, intensity_matrix, coordinates, ion_mode = process_ms_imaging_data(
        file_path,
        os.path.splitext(file_path)[0] + '.ibd',
        mz_bin_size=mz_bin_size,
        mass_detect_int_tol=mass_detect_int_tol,
        noise_detection=noise_detection,
        sn_factor=sn_factor,
        centroided=centroided,
        n_processes=n_processes,
        save=True, save_dir=result_folder
    )

    print(f"Calculating ion image correlations for {file_name}")
    cor_matrix = calc_all_mz_correlations(intensity_matrix,
                                          min_overlap=min_overlap,
                                          min_cor=min_correlation,
                                          n_processes=n_processes,
                                          save=True,
                                          save_dir=result_folder)

    print(f"Generating pseudo MS2 spectra for {file_name}")
    pseudo_ms2 = generate_pseudo_ms2(mz_values, intensity_matrix, cor_matrix,
                                     n_processes=n_processes,
                                     min_cluster_size=ms1id_min_matched_peak + 1,
                                     min_cor=min_correlation,
                                     max_cor_depth=max_cor_depth,
                                     save=True,
                                     save_dir=result_folder)

    print(f"Annotating pseudo MS2 spectra for {file_name}")
    pseudo_ms2 = ms1_id_annotation(pseudo_ms2, library_path, n_processes=None,
                                   mz_tol=library_search_mztol,
                                   ion_mode=ion_mode,
                                   score_cutoff=ms1id_score_cutoff,
                                   min_matched_peak=ms1id_min_matched_peak,
                                   min_spec_usage=ms1id_min_spec_usage,
                                   max_prec_rel_int_in_other_ms2=max_prec_rel_int_in_other_ms2,
                                   save=True,
                                   save_dir=result_folder)

    print(f"Writing results for {file_name}")
    write_ms1_id_results(pseudo_ms2, save=True, save_dir=result_folder)

    return
