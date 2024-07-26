from ms1id_lcms_workflow import main_workflow_single, main_workflow
import os
import multiprocessing
from functools import partial


def ms1id_single_file(file_path, library_path):
    print('===========================================')
    print(f'Processing {file_path}')

    main_workflow_single(
        file_path=file_path,
        msms_library_path=library_path,
        ms1_id=True, ms2_id=True,
        mz_tol_ms1=0.01, mz_tol_ms2=0.015,
        mass_detect_int_tol=10000,
        peak_cor_rt_tol=0.025,
        min_ppc=0.9, roi_min_length=4,
        ms1id_score_cutoff=0.7, ms1id_min_matched_peak=4,
        ms1id_min_prec_int_in_ms1=1e5,
        ms1id_max_prec_rel_int_in_other_ms2=0.01,
        ms2id_score_cutoff=0.7, ms2id_min_matched_peak=4,
        plot_bpc=False)


def ms1id_single_file_batch(file_dir, library_path, num_processes=None):
    """
    Process all files in a directory in single file mode using parallel processing.
    """
    # Get all mzXML and mzML files in the directory
    files = [f for f in os.listdir(file_dir) if f.endswith('.mzXML') or f.endswith('.mzML')]
    files = [os.path.join(file_dir, f) for f in files]

    # If num_processes is not specified, use the number of CPU cores
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Create a partial function with the library_path argument
    process_file = partial(ms1id_single_file, library_path=library_path)

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the process_file function to all files
        pool.map(process_file, files)


if __name__ == '__main__':

    ms1id_single_file_batch(file_dir='../data/from_vincent/data',
                            library_path='../data/gnps_nist20.pkl')

    # ms1id_single_file(
    #     file_path='/Users/shipei/Documents/projects/ms1_id/data/trial_data/single/Standards_p_1ugmL_glycocholic.mzXML',
    #     library_path='/Users/shipei/Documents/projects/ms1_id/data/gnps_nist20.pkl')

    # GNPS+NIST: /Users/shipei/Documents/projects/ms1_id/data/gnps_nist20.pkl
    # GNPS: /Users/shipei/Documents/projects/ms1_id/data/ALL_GNPS_NO_PROPOGATED.pkl
    # NIST: /Users/shipei/Documents/projects/ms1_id/data/nist20.pkl