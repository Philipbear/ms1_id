import os
import pickle
from typing import List
import numpy as np
from multiprocessing import Pool, cpu_count
from ms_entropy import read_one_spectrum
from tqdm import tqdm

from _utils_imaging import SpecAnnotation
from flash_cos import FlashCos


def prepare_ms2_lib(ms2db, mz_tol=0.02, peak_scale_k=10, peak_intensity_power=0.5):
    """
    prepare ms2 db using MSP formatted database
    :return: a pickle file
    """
    replace_keys = {'precursormz': 'precursor_mz',
                    'precursortype': 'precursor_type',
                    'ionmode': 'ion_mode',
                    # 'instrumenttype': 'instrument_type',
                    'collisionenergy': 'collision_energy'}

    db = []
    for a in read_one_spectrum(ms2db):
        # replace keys if exist
        for k, v in replace_keys.items():
            if k in a:
                a[v] = a.pop(k)

        # convert precursor_mz to float
        try:
            a['precursor_mz'] = float(a['precursor_mz'])
        except:
            a['precursor_mz'] = 0.0

        # ion_mode, harmonize
        if 'ion_mode' in a:
            a['ion_mode'] = a['ion_mode'].lower().strip()
            if a['ion_mode'] in ['p', 'positive']:
                a['ion_mode'] = 'positive'
            elif a['ion_mode'] in ['n', 'negative']:
                a['ion_mode'] = 'negative'
            else:
                a['ion_mode'] = 'unknown'

        db.append(a)

    print('Number of spectra in the database:', len(db))

    print('initializing search engine')
    search_engine = FlashCos(max_ms2_tolerance_in_da=mz_tol * 1.005,
                             mz_index_step=0.0001,
                             peak_scale_k=peak_scale_k,
                             peak_intensity_power=peak_intensity_power)
    print('building index')
    search_engine.build_index(db,
                              max_indexed_mz=2000,
                              precursor_ions_removal_da=0.5,
                              noise_threshold=0.0,
                              min_ms2_difference_in_da=mz_tol * 2.02,
                              clean_spectra=True)

    if peak_scale_k is None:
        new_path = os.path.splitext(ms2db)[0] + '.pkl'
    else:
        new_path = os.path.splitext(ms2db)[0] + f'_k{peak_scale_k}.pkl'
    # save as pickle
    with open(new_path, 'wb') as file:
        # Dump the data into the file
        pickle.dump(search_engine, file)

    print(f"Pickle file saved to: {new_path}")
    return search_engine


def ms1_id_annotation(ms1_spec_ls, ms2_library, n_processes=None,
                      mz_tol=0.01,
                      score_cutoff=0.6, min_matched_peak=4,
                      ion_mode=None,
                      save=False, save_dir=None,
                      chunk_size=1000):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param n_processes: number of processes to use
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :param peak_scale_k: for rev cos, peak scaling factor
    :param ion_mode: str, ion mode. If None, all ion modes are considered
    :param save: bool, whether to save the results
    :param save_dir: str, directory to save the results
    :param chunk_size: int, number of spectra to process in each parallel task
    :return: PseudoMS1-like object
    """

    # check if results are already annotated
    if save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms1_annotated.pkl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                ms1_spec_ls = pickle.load(file)
            return ms1_spec_ls

    if n_processes is None:
        n_processes = max(1, cpu_count() // 5)  # ms2 library is large, for RAM usage

    chunk_size = min(chunk_size, len(ms1_spec_ls))

    # perform revcos matching
    ms1_spec_ls = ms1_id_revcos_matching(ms1_spec_ls, ms2_library, n_processes=n_processes,
                                         mz_tol=mz_tol,
                                         ion_mode=ion_mode,
                                         score_cutoff=score_cutoff,
                                         min_matched_peak=min_matched_peak,
                                         chunk_size=chunk_size)

    if save:
        save_path = os.path.join(save_dir, 'pseudo_ms1_annotated.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(ms1_spec_ls, file)

    return ms1_spec_ls


def ms1_id_revcos_matching(ms1_spec_ls: List, ms2_library: str, n_processes: int = None,
                           mz_tol: float = 0.02,
                           ion_mode: str = None,
                           score_cutoff: float = 0.7,
                           min_matched_peak: int = 3,
                           chunk_size: int = 500) -> List:
    """
    Perform MS1 annotation using parallel open search for the entire spectrum, with filters similar to identity search.

    :param ms1_spec_ls: a list of PseudoMS1-like objects
    :param ms2_library: path to the pickle file, indexed library
    :param n_processes: number of processes to use
    :param mz_tol: m/z tolerance in Da, for open matching
    :param ion_mode: str, ion mode
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :param chunk_size: number of spectra to process in each parallel task
    :return: List of updated PseudoMS1-like objects
    """
    mz_tol = max(mz_tol, 0.02)  # indexed library mz_tol is 0.02

    # Load the data
    with open(ms2_library, 'rb') as file:
        search_eng = pickle.load(file)

    # Prepare chunks
    chunks = [ms1_spec_ls[i:i + chunk_size] for i in range(0, len(ms1_spec_ls), chunk_size)]

    # Prepare arguments for parallel processing
    args_list = [(chunk, search_eng, mz_tol, ion_mode, score_cutoff, min_matched_peak)
                 for chunk in chunks]

    # Use multiprocessing to process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_chunk, args_list), total=len(args_list), desc="Processing chunks"))

    # Flatten results
    return [spec for chunk_result in results for spec in chunk_result]


def _process_chunk(args):
    chunk, search_eng, mz_tol, ion_mode, score_cutoff, min_matched_peak = args

    for spec in chunk:
        matching_result = search_eng.search(
            precursor_mz=spec.t_mz,
            peaks=[[mz, intensity] for mz, intensity in zip(spec.mzs, spec.intensities)],
            ms1_tolerance_in_da=mz_tol,
            ms2_tolerance_in_da=mz_tol,
            method="identity",
            precursor_ions_removal_da=0.5,
            noise_threshold=0.0,
            min_ms2_difference_in_da=mz_tol * 2.02,
            reverse=True
        )

        score_arr, matched_peak_arr, spec_usage_arr = matching_result['identity_search']

        # filter by matching cutoffs, including scaled score
        v = np.where(np.logical_and(score_arr >= score_cutoff, matched_peak_arr >= min_matched_peak))[0]

        all_matches = []
        for idx in v:
            matched = {k.lower(): v for k, v in search_eng[idx].items()}

            this_ion_mode = matched.get('ion_mode', '')
            if ion_mode is not None and ion_mode != this_ion_mode:
                continue

            all_matches.append((idx, score_arr[idx], matched_peak_arr[idx], spec_usage_arr[idx]))

        if all_matches:
            spec.annotated = True
            spec.annotation_ls = []
            for idx, score, matched_peaks, spectral_usage in all_matches:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}

                annotation = SpecAnnotation(idx, score, matched_peaks)
                annotation.spectral_usage = spectral_usage

                annotation.name = matched.get('name', '')
                annotation.precursor_mz = matched.get('precursor_mz')
                annotation.precursor_type = matched.get('precursor_type', None)
                annotation.formula = matched.get('formula', None)
                annotation.inchikey = matched.get('inchikey', None)
                annotation.instrument_type = matched.get('instrument_type', None)
                annotation.collision_energy = matched.get('collision_energy', None)
                annotation.peaks = matched.get('peaks', None)
                annotation.db_id = matched.get('comment', None)
                annotation.matched_spec = matched.get('peaks', None)

                spec.annotation_ls.append(annotation)

            # sort the annotations by precursor m/z in descending order
            spec.annotation_ls = sorted(spec.annotation_ls, key=lambda x: x.precursor_mz, reverse=True)
        else:
            spec.annotated = False

    return chunk

