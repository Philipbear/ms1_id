import os
from _process_msi_data import process_ms_imaging_data
from _calculate_mz_cor_parallel import calc_all_mz_correlations
from _group_mz_cor_parallel import generate_pseudo_ms1
from _reverse_matching_parallel import ms1_id_annotation
from _export_msi import write_ms1_id_results


def ms1id_imaging_single_workflow(file_path, msms_library_path, n_processes=None,
                                  mass_detect_int_tol=None, max_mz=None,
                                  mz_bin_size=0.01,
                                  min_overlap=5, min_correlation=0.9, min_cluster_size=6,
                                  ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                                  ms1id_min_prec_sn_ratio=3, ms1id_max_prec_rel_int_in_other_ms2=0.05):
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace('.imzML', '')

    # make a result folder of file_name
    result_folder = os.path.join(file_dir, file_name)
    os.makedirs(result_folder, exist_ok=True)

    print(f"Processing {file_name}")
    mz_values, intensity_matrix, coordinates, ion_mode, actual_mass_detect_int_tol = process_ms_imaging_data(
        file_path,
        file_path.replace('.imzML', '.ibd'),
        mass_detect_int_tol=mass_detect_int_tol,
        max_mz=max_mz,
        mz_bin_size=mz_bin_size,
        save=True, save_dir=result_folder
    )

    print(f"Calculating ion image correlations for {file_name}")
    cor_matrix = calc_all_mz_correlations(intensity_matrix,
                                          min_overlap=min_overlap,
                                          min_cor=min_correlation,
                                          n_processes=n_processes,
                                          save=True,
                                          save_dir=result_folder)

    print(f"Generating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = generate_pseudo_ms1(mz_values, intensity_matrix, cor_matrix,
                                     n_processes=n_processes,
                                     min_cluster_size=min_cluster_size,
                                     save=True,
                                     save_dir=result_folder)

    print(f"Annotating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = ms1_id_annotation(pseudo_ms1, msms_library_path, n_processes=None,
                                   mz_tol=ms1id_mz_tol,
                                   ion_mode=ion_mode,
                                   score_cutoff=ms1id_score_cutoff, min_matched_peak=ms1id_min_matched_peak,
                                   min_prec_int_in_ms1=ms1id_min_prec_sn_ratio * actual_mass_detect_int_tol,
                                   max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                                   save=True,
                                   save_dir=result_folder)

    print(f"Writing results for {file_name}")
    write_ms1_id_results(pseudo_ms1, save=True, save_dir=result_folder)

    return


if __name__ == '__main__':
    ############################
    # Single workflow
    file_path = '../../imaging/MTBLS313/Brain01_Bregma-3-88b_centroid.imzML'
    ms1id_imaging_single_workflow(file_path=file_path,
                                  msms_library_path='../../data/gnps.pkl',
                                  mass_detect_int_tol=None, max_mz=None, mz_bin_size=0.01,
                                  min_overlap=5, min_correlation=0.9, min_cluster_size=5,
                                  ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                                  ms1id_min_prec_sn_ratio=3, ms1id_max_prec_rel_int_in_other_ms2=0.05)
