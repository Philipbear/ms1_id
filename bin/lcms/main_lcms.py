from ms1id_lcms_workflow import main_workflow_single, main_workflow
import os
import multiprocessing
from functools import partial


def ms1id_single_file(file_path, library_path,
                      ms1_id=True, ms2_id=False,
                      mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                      mass_detect_int_tol=10000,
                      peak_cor_rt_tol=0.025,
                      min_ppc=0.9, roi_min_length=4,
                      ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                      ms1id_min_prec_int_in_ms1=1e5,
                      ms1id_max_prec_rel_int_in_other_ms2=0.01,
                      ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                      out_dir=None):
    print('===========================================')
    print(f'Processing {file_path}')

    main_workflow_single(
        file_path=file_path,
        msms_library_path=library_path,
        ms1_id=ms1_id, ms2_id=ms2_id,
        mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
        mass_detect_int_tol=mass_detect_int_tol,
        peak_cor_rt_tol=peak_cor_rt_tol,
        min_ppc=min_ppc, roi_min_length=roi_min_length,
        ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
        ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
        ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
        ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
        out_dir=out_dir)

    return


def ms1id_single_file_batch(data_dir, library_path,
                            parallel=True, num_processes=None,
                            ms1_id=True, ms2_id=False,
                            mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                            mass_detect_int_tol=10000,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.9, roi_min_length=4,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                            ms1id_min_prec_int_in_ms1=1e5,
                            ms1id_max_prec_rel_int_in_other_ms2=0.01,
                            ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                            out_dir=None
                            ):
    """
    Process all files in a directory in single file mode using parallel processing.
    """
    # Get all mzXML and mzML files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.mzXML') or f.endswith('.mzML')]
    files = [os.path.join(data_dir, f) for f in files]

    if parallel:
        # If num_processes is not specified, use the number of CPU cores
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Create a partial function with the library_path argument
        process_file = partial(ms1id_single_file, library_path=library_path,
                               ms1_id=ms1_id, ms2_id=ms2_id,
                               mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
                               mass_detect_int_tol=mass_detect_int_tol,
                               peak_cor_rt_tol=peak_cor_rt_tol,
                               min_ppc=min_ppc, roi_min_length=roi_min_length,
                               ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                               ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                               ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                               ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
                               out_dir=out_dir)

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Map the process_file function to all files
            pool.map(process_file, files)
    else:
        # Process each file sequentially
        for file in files:
            ms1id_single_file(file, library_path,
                              ms1_id=ms1_id, ms2_id=ms2_id,
                              mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
                              mass_detect_int_tol=mass_detect_int_tol,
                              peak_cor_rt_tol=peak_cor_rt_tol,
                              min_ppc=min_ppc, roi_min_length=roi_min_length,
                              ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                              ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                              ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                              ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
                              out_dir=out_dir)

    return


def ms1id_batch_workflow(project_dir, library_path, sample_dir='data',
                         batch_size=100, cpu_ratio=0.9,
                         ms1_id=True, ms2_id=False,
                         mz_tol_ms1=0.01, mz_tol_ms2=0.015,
                         mass_detect_int_tol=10000,
                         peak_cor_rt_tol=0.025,
                         min_ppc=0.9, roi_min_length=4,
                         ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                         ms1id_min_prec_int_in_ms1=1e5,
                         ms1id_max_prec_rel_int_in_other_ms2=0.01,
                         ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3):
    main_workflow(project_path=project_dir,
                  msms_library_path=library_path,
                  sample_dir=sample_dir,
                  batch_size=batch_size, cpu_ratio=cpu_ratio,
                  ms1_id=ms1_id, ms2_id=ms2_id,
                  mz_tol_ms1=mz_tol_ms1, mz_tol_ms2=mz_tol_ms2,
                  mass_detect_int_tol=mass_detect_int_tol,
                  peak_cor_rt_tol=peak_cor_rt_tol,
                  min_ppc=min_ppc, roi_min_length=roi_min_length,
                  ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                  ms1id_min_prec_int_in_ms1=ms1id_min_prec_int_in_ms1,
                  ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                  ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak)

    return


if __name__ == '__main__':
    # ms1id_single_file(
    #     file_path='/Users/shipei/Documents/projects/ms1_id/data/trial_data/single/Standards_p_1ugmL_glycocholic.mzXML',
    #     library_path='/Users/shipei/Documents/projects/ms1_id/data/gnps_nist20.pkl')

    data_dirs = ['../../data/PR000677_data/c8_pos', '../../data/PR000677_data/c18_neg',
                 '../../data/PR000677_data/hilic_pos', '../../data/PR000677_data/hilic_neg']
    for data_dir in data_dirs:
        ms1id_single_file_batch(data_dir=data_dir,
                                library_path='../../data/gnps_nist20.pkl',
                                parallel=True, num_processes=None,
                                ms1_id=True, ms2_id=False,
                                mz_tol_ms1=0.015, mz_tol_ms2=0.02,
                                mass_detect_int_tol=30000,
                                peak_cor_rt_tol=0.05,
                                min_ppc=0.8, roi_min_length=6,
                                ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
                                ms1id_min_prec_int_in_ms1=1e5,
                                ms1id_max_prec_rel_int_in_other_ms2=0.01,
                                ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                                out_dir=data_dir.replace('PR000677_data', 'PR000677_output'))

    # ms1id_batch_workflow(project_dir='../../data/test',
    #                      library_path='../../data/gnps_nist20.pkl',
    #                      sample_dir='data',
    #                      batch_size=100, cpu_ratio=0.9,
    #                      ms1_id=True, ms2_id=True,
    #                      mz_tol_ms1=0.015, mz_tol_ms2=0.02,
    #                      mass_detect_int_tol=30000,
    #                      peak_cor_rt_tol=0.05,
    #                      min_ppc=0.9, roi_min_length=4,
    #                      ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3,
    #                      ms1id_min_prec_int_in_ms1=1e5,
    #                      ms1id_max_prec_rel_int_in_other_ms2=0.01,
    #                      ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3)
