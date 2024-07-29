import os
from _process_imgaing_data import process_ms_imaging_data
from _calculate_mz_cor import calc_all_mz_correlations
from _group_mz_cor import generate_pseudo_ms1
from _reverse_matching import ms1_id_annotation
from _export_imaging import write_ms1_id_results


def ms1id_imaging_single_workflow(file_path, msms_library_path,
                                  mass_detect_int_tol=None, mz_bin_size=0.01,
                                  min_spec_overlap_ratio=0.1, min_correlation=0.8, min_cluster_size=6,
                                  ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                                  ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05):
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace('.imzML', '')

    # make a result folder of file_name
    result_folder = os.path.join(file_dir, file_name)
    os.makedirs(result_folder, exist_ok=True)

    print(f"Processing {file_name}")
    mz_values, intensity_matrix, coordinates = process_ms_imaging_data(file_path, file_path.replace('.imzML', '.ibd'),
                                                                       mass_detect_int_tol=mass_detect_int_tol,
                                                                       mz_bin_size=mz_bin_size,
                                                                       save=True, save_dir=result_folder)

    print(f"Calculating ion image correlations for {file_name}")
    cor_matrix = calc_all_mz_correlations(intensity_matrix,
                                          min_spec_overlap_ratio=min_spec_overlap_ratio,
                                          save=True,
                                          save_dir=result_folder)

    print(f"Generating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = generate_pseudo_ms1(mz_values, intensity_matrix, cor_matrix,
                                     min_correlation=min_correlation,
                                     min_cluster_size=min_cluster_size,
                                     save=True,
                                     save_dir=result_folder)

    print(f"Annotating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = ms1_id_annotation(pseudo_ms1, msms_library_path, mz_tol=ms1id_mz_tol,
                                   score_cutoff=ms1id_score_cutoff, min_matched_peak=ms1id_min_matched_peak,
                                   min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                                   max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)

    print(f"Writing results for {file_name}")
    write_ms1_id_results(pseudo_ms1, save=True, save_dir=result_folder)

    return


if __name__ == '__main__':
    ############################
    # Single workflow
    file_path = '../../imaging/MTBLS313/Brain01_Bregma-3-88b_centroid.imzML'
    ms1id_imaging_single_workflow(file_path=file_path,
                                  msms_library_path='../../data/gnps_nist20.pkl',
                                  mass_detect_int_tol=None, mz_bin_size=0.01,
                                  min_spec_overlap_ratio=0.5, min_correlation=0.9, min_cluster_size=5,
                                  ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                                  ms1id_min_prec_int_in_ms1=0, ms1id_max_prec_rel_int_in_other_ms2=0.05)

