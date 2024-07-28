import os
from _process_imgaing_data import process_ms_imaging_data
from _calculate_mz_cor import calc_all_mz_correlations
from _group_mz_cor import generate_pseudo_ms1
from _reverse_matching import ms1_id_annotation
from _export_imaging import write_ms1_id_results


def ms1id_imaging_workflow(file_path, msms_library_path,
                           mass_detect_int_tol=500.0, bin_size=0.01,
                           min_spec_overlap_ratio=0.1, min_correlation=0.8, min_cluster_size=6,
                           ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                           ms1id_min_prec_int_in_ms1=500, ms1id_max_prec_rel_int_in_other_ms2=0.05):

    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path).replace('.imzML', '')

    ibd_file = file_path.replace('.imzML', '.ibd')

    print(f"Processing {file_name}")
    mz_values, intensity_matrix, coordinates = process_ms_imaging_data(file_path, ibd_file,
                                                                       mass_detect_int_tol=mass_detect_int_tol,
                                                                       bin_size=bin_size,
                                                                       save=True)

    print(f"Calculating ion image correlations for {file_name}")
    cor_matrix = calc_all_mz_correlations(intensity_matrix,
                                          min_spec_overlap_ratio=min_spec_overlap_ratio,
                                          save=True,
                                          path=os.path.join(file_dir, f'{file_name}_cor_matrix.npz'))

    print(f"Generating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = generate_pseudo_ms1(mz_values, intensity_matrix, cor_matrix,
                                     min_correlation=min_correlation,
                                     min_cluster_size=min_cluster_size,
                                     save=True,
                                     save_path=os.path.join(file_dir, f'{file_name}_pseudo_ms1.pkl'))

    print(f"Annotating pseudo MS1 spectra for {file_name}")
    pseudo_ms1 = ms1_id_annotation(pseudo_ms1, msms_library_path, mz_tol=ms1id_mz_tol,
                                   score_cutoff=ms1id_score_cutoff, min_matched_peak=ms1id_min_matched_peak,
                                   min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                                   max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2)

    print(f"Writing results for {file_name}")
    write_ms1_id_results(pseudo_ms1, save=True, save_path=os.path.join(file_dir, f'{file_name}_annotations.tsv'))

    return


if __name__ == '__main__':
    ms1id_imaging_workflow(file_path='../../imaging/MTBLS313/Brain01_Bregma-3-88b_centroid.imzML',
                           msms_library_path='../../data/gnps.pkl',
                           mass_detect_int_tol=1000, bin_size=0.01,
                           min_spec_overlap_ratio=0.5, min_correlation=0.8, min_cluster_size=5,
                           ms1id_mz_tol=0.01, ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
                           ms1id_min_prec_int_in_ms1=1000, ms1id_max_prec_rel_int_in_other_ms2=0.05)

    # load
    import numpy as np
    import pickle
    # mz_values = np.load('../../imaging/bottom_control_1_mz_values.npy')
    # intensity_matrix = np.load('../../imaging/bottom_control_1_intensity_matrix.npy')
    # pseudo_ms1 = pickle.load(open('../../imaging/bottom_control_1_pseudo_ms1.pkl', 'rb'))
    # print('loaded')



