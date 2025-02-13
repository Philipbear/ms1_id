from ms1_id.lcms.ms1id_lcms_single import main_workflow_single
from ms1_id.lcms.ms1id_lcms_batch import main_workflow
import os
import concurrent.futures
from functools import partial


def ms1id_single_file(file_path, ms1id_library_path=None, ms2id_library_path=None,
                      ms1_id=True, ms2_id=False,
                      ms1_tol=0.01, ms2_tol=0.015,
                      mass_detect_int_tol=10000,
                      peak_cor_rt_tol=0.025,
                      min_ppc=0.8, roi_min_length=4,
                      library_search_mztol=0.05,
                      ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3, ms1id_min_spec_usage=0.05,
                      ms1id_max_prec_rel_int_in_other_ms2=0.01,
                      ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3,
                      out_dir=None):
    print('===========================================')
    print(f'Processing {file_path}')

    main_workflow_single(
        file_path=file_path,
        ms1id_library_path=ms1id_library_path, ms2id_library_path=ms2id_library_path,
        ms1_id=ms1_id, ms2_id=ms2_id,
        ms1_tol=ms1_tol, ms2_tol=ms2_tol,
        mass_detect_int_tol=mass_detect_int_tol,
        peak_cor_rt_tol=peak_cor_rt_tol,
        min_ppc=min_ppc, roi_min_length=roi_min_length,
        library_search_mztol=library_search_mztol,
        ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
        ms1id_min_spec_usage=ms1id_min_spec_usage,
        ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
        ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
        out_dir=out_dir)

    return


def ms1id_single_file_batch(data_dir, ms1id_library_path=None, ms2id_library_path=None,
                            parallel=True, num_processes=None,
                            ms1_id=True, ms2_id=False,
                            ms1_tol=0.01, ms2_tol=0.015,
                            mass_detect_int_tol=10000,
                            peak_cor_rt_tol=0.025,
                            min_ppc=0.8, roi_min_length=4,
                            library_search_mztol=0.05,
                            ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3, ms1id_min_spec_usage=0.05,
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

    # if some file results exist in out_dir, skip them
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in files if not os.path.exists(
            os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '_feature_table.tsv'))]

    if len(files) == 0:
        print(f"No files to process in {data_dir}")
        return

    # Create a partial function with the library_path argument
    process_file = partial(ms1id_single_file,
                           ms1id_library_path=ms1id_library_path, ms2id_library_path=ms2id_library_path,
                           ms1_id=ms1_id, ms2_id=ms2_id,
                           ms1_tol=ms1_tol, ms2_tol=ms2_tol,
                           mass_detect_int_tol=mass_detect_int_tol,
                           peak_cor_rt_tol=peak_cor_rt_tol,
                           min_ppc=min_ppc, roi_min_length=roi_min_length,
                           library_search_mztol=library_search_mztol,
                           ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                           ms1id_min_spec_usage=ms1id_min_spec_usage,
                           ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                           ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
                           out_dir=out_dir)

    if parallel:
        # If num_processes is not specified, use the number of CPU cores
        if num_processes is None:
            num_processes = min(os.cpu_count(), len(files))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            futures = [executor.submit(process_file, file) for file in files]

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred during execution
                except Exception as e:
                    print(f"An error occurred: {e}")
    else:
        # Process each file sequentially
        for file in files:
            ms1id_single_file(file_path=file,
                              ms1id_library_path=ms1id_library_path, ms2id_library_path=ms2id_library_path,
                              ms1_id=ms1_id, ms2_id=ms2_id,
                              ms1_tol=ms1_tol, ms2_tol=ms2_tol,
                              mass_detect_int_tol=mass_detect_int_tol,
                              peak_cor_rt_tol=peak_cor_rt_tol,
                              min_ppc=min_ppc, roi_min_length=roi_min_length,
                              library_search_mztol=library_search_mztol,
                              ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                              ms1id_min_spec_usage=ms1id_min_spec_usage,
                              ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                              ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak,
                              out_dir=out_dir)

    return


def ms1id_batch_mode(project_path=None, ms1id_library_path=None, ms2id_library_path=None,
                     sample_dir='data', parallel=True,
                     ms1_id=True, ms2_id=False,
                     cpu_ratio=0.8,
                     run_rt_correction=True, run_normalization=True,
                     ms1_tol=0.01, ms2_tol=0.015, mass_detect_int_tol=None,
                     align_mz_tol=0.015, align_rt_tol=0.2, alignment_drop_by_fill_pct_ratio=0.1,
                     peak_cor_rt_tol=0.025,
                     min_ppc=0.8, roi_min_length=5, library_search_mztol=0.05,
                     ms1id_score_cutoff=0.7, ms1id_min_matched_peak=3, ms1id_min_spec_usage=0.05,
                     ms1id_max_prec_rel_int_in_other_ms2=0.01,
                     ms2id_score_cutoff=0.7, ms2id_min_matched_peak=3):
    main_workflow(project_path=project_path, ms1id_library_path=ms1id_library_path,
                  ms2id_library_path=ms2id_library_path,
                  sample_dir=sample_dir,
                  parallel=parallel, ms1_id=ms1_id, ms2_id=ms2_id,
                  cpu_ratio=cpu_ratio,
                  run_rt_correction=run_rt_correction, run_normalization=run_normalization,
                  ms1_tol=ms1_tol, ms2_tol=ms2_tol, mass_detect_int_tol=mass_detect_int_tol,
                  align_mz_tol=align_mz_tol, align_rt_tol=align_rt_tol,
                  alignment_drop_by_fill_pct_ratio=alignment_drop_by_fill_pct_ratio,
                  peak_cor_rt_tol=peak_cor_rt_tol, min_ppc=min_ppc, roi_min_length=roi_min_length,
                  library_search_mztol=library_search_mztol,
                  ms1id_score_cutoff=ms1id_score_cutoff, ms1id_min_matched_peak=ms1id_min_matched_peak,
                  ms1id_min_spec_usage=ms1id_min_spec_usage,
                  ms1id_max_prec_rel_int_in_other_ms2=ms1id_max_prec_rel_int_in_other_ms2,
                  ms2id_score_cutoff=ms2id_score_cutoff, ms2id_min_matched_peak=ms2id_min_matched_peak)

    return
