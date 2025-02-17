import os
import pickle
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
from tqdm import tqdm

from ms1_id.msi.centroid_data import centroid_spectrum_for_search
from ms1_id.msi.utils_imaging import SpecAnnotation
from ms1_id.search_utils.flash_cos import FlashCos


def validate_library_path(library_path):
    """
    Validate the library path
    :param library_path: str or list of str
    :return: list of str
    """
    # if library_ls is a string, convert to list
    if isinstance(library_path, str):
        library_path = [library_path]

    for path in library_path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Library file not found: {path}")

    return library_path


def ms1_id_annotation(ms1_spec_ls, library_ls, n_processes=None,
                      mz_tol=0.05,
                      score_cutoff=0.6, min_matched_peak=4, min_spec_usage=0.0,
                      ion_mode=None,
                      refine=False,
                      max_prec_rel_int_in_other_ms2=0.05,
                      save=False, save_dir=None,
                      chunk_size=1000):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS2-like object
    :param library: path to the pickle file, indexed library
    :param n_processes: number of processes to use
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :param peak_scale_k: for rev cos, peak scaling factor
    :param ion_mode: str, ion mode. If None, all ion modes are considered
    :param refine: bool, whether to refine the results
    :param max_prec_rel_int_in_other_ms2: float, maximum relative intensity of precursor in other MS2 spectra
    :param save: bool, whether to save the results
    :param save_dir: str, directory to save the results
    :param chunk_size: int, number of spectra to process in each parallel task
    :return: PseudoMS2-like object
    """

    # check if results are already annotated
    if save_dir:
        save_path = os.path.join(save_dir, 'pseudo_ms2_annotated.pkl')
        if os.path.exists(save_path):
            with open(save_path, 'rb') as file:
                ms1_spec_ls = pickle.load(file)
            return ms1_spec_ls

    if n_processes is None:
        n_processes = max(1, cpu_count() // 5)  # ms2 library is large, for RAM usage

    chunk_size = min(chunk_size, len(ms1_spec_ls))

    # Perform centroiding for all spectra before annotation
    ms1_spec_ls = centroid_all_spectra(ms1_spec_ls, n_processes)

    # perform revcos matching
    ms1_spec_ls = ms1_id_revcos_matching(ms1_spec_ls, library_ls, n_processes=n_processes,
                                         mz_tol=mz_tol,
                                         ion_mode=ion_mode,
                                         score_cutoff=score_cutoff,
                                         min_matched_peak=min_matched_peak,
                                         min_spec_usage=min_spec_usage,
                                         chunk_size=chunk_size)

    # refine the results, to avoid wrong annotations (ATP, ADP, AMP all annotated at the same RT)
    if refine:
        print('Refining MS1 ID results...')
        ms1_spec_ls = refine_ms1_id_results(ms1_spec_ls, mz_tol=mz_tol,
                                            max_prec_rel_int=max_prec_rel_int_in_other_ms2)

    if save:
        save_path = os.path.join(save_dir, 'pseudo_ms2_annotated.pkl')
        with open(save_path, 'wb') as file:
            pickle.dump(ms1_spec_ls, file)

    return ms1_spec_ls


def centroid_all_spectra(ms1_spec_ls, n_processes):
    """
    Centroid all spectra in parallel
    """
    with Pool(processes=n_processes) as pool:
        ms1_spec_ls = list(tqdm(pool.imap(_centroid_spectrum, ms1_spec_ls),
                                total=len(ms1_spec_ls),
                                desc="Centroiding spectra"))
    return ms1_spec_ls


def _centroid_spectrum(spec):
    peaks = list(zip(spec.mzs, spec.intensities))
    peaks = np.asarray(peaks, dtype=np.float32, order="C")
    spec.centroided_peaks = centroid_spectrum_for_search(peaks, width_da=0.05 * 2.015)
    return spec


def ms1_id_revcos_matching(ms1_spec_ls, library_ls, n_processes=None,
                           mz_tol=0.05,
                           ion_mode=None,
                           score_cutoff=0.7,
                           min_matched_peak=4,
                           min_spec_usage=0.0,
                           chunk_size=500) -> List:
    """
    Perform MS1 annotation using parallel open search for the entire spectrum, with filters similar to identity search.

    :param ms1_spec_ls: a list of PseudoMS2-like objects
    :param library_ls: path to the pickle file, indexed library
    :param n_processes: number of processes to use
    :param mz_tol: m/z tolerance in Da, for open matching
    :param ion_mode: str, ion mode
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :param min_spec_usage: minimum spectral usage
    :param chunk_size: number of spectra to process in each parallel task
    :return: List of updated PseudoMS2-like objects
    """
    mz_tol = min(mz_tol, 0.05)  # indexed library mz_tol is 0.05

    # Load all libraries
    search_engines = []
    for library in library_ls:
        with open(library, 'rb') as file:
            search_eng = pickle.load(file)
        db_name = os.path.basename(library)
        search_engines.append((search_eng, db_name))
        print(f"Loaded library: {db_name}")

    # Prepare chunks
    chunks = [ms1_spec_ls[i:i + chunk_size] for i in range(0, len(ms1_spec_ls), chunk_size)]

    # Prepare arguments for parallel processing
    args_list = [(chunk, search_engines, mz_tol, ion_mode, score_cutoff, min_matched_peak, min_spec_usage)
                 for chunk in chunks]

    # Use multiprocessing to process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_chunk_multi_lib, args_list), total=len(args_list), desc="Annotation: "))

    # Flatten results
    return [spec for chunk_result in results for spec in chunk_result]


def _process_chunk_multi_lib(args):
    chunk, search_engines, mz_tol, ion_mode, score_cutoff, min_matched_peak, min_spec_usage = args

    for spec in chunk:
        for search_eng, db_name in search_engines:

            if len(spec.centroided_peaks) < min_matched_peak:
                continue

            # open search
            matching_result = search_eng.search(
                precursor_mz=2000.00,  # unused, open search
                peaks=spec.centroided_peaks,
                ms1_tolerance_in_da=mz_tol,
                ms2_tolerance_in_da=mz_tol,
                method="open",
                precursor_ions_removal_da=0.5,
                noise_threshold=0.0,
                min_ms2_difference_in_da=mz_tol * 2.02,
                reverse=True
            )

            score_arr, matched_peak_arr, spec_usage_arr = matching_result['open_search']

            # filter by matching cutoffs
            v = np.where((score_arr >= score_cutoff) &
                         (matched_peak_arr >= min_matched_peak) &
                         (spec_usage_arr >= min_spec_usage))[0]

            all_matches = []
            nonzero_mask = np.array(spec.intensities) > 0
            nonzero_mzs = np.array(spec.mzs)[nonzero_mask]
            for idx in v:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}

                this_ion_mode = matched.get('ion_mode', '')
                if ion_mode is not None and ion_mode != this_ion_mode:
                    continue

                # precursor should be in the pseudo MS2 spectrum with intensity > 0
                precursor_mz = matched.get('precursor_mz', 0)
                if not any(np.isclose(nonzero_mzs, precursor_mz, atol=mz_tol)):
                    continue

                all_matches.append((idx, score_arr[idx], matched_peak_arr[idx], spec_usage_arr[idx]))

            if all_matches:
                spec.annotated = True
                for idx, score, matched_peaks, spectral_usage in all_matches:
                    matched = {k.lower(): v for k, v in search_eng[idx].items()}

                    annotation = SpecAnnotation(db_name, idx, score, matched_peaks)
                    annotation.spectral_usage = spectral_usage

                    annotation.name = matched.get('name', '')

                    matched_spec_precursor_mz = matched.get('precursor_mz')
                    annotation.precursor_mz = matched_spec_precursor_mz
                    annotation.mz = min(nonzero_mzs, key=lambda x: abs(x - matched_spec_precursor_mz))

                    annotation.precursor_type = matched.get('precursor_type', None)
                    annotation.formula = matched.get('formula', None)
                    annotation.inchikey = matched.get('inchikey', None)
                    annotation.instrument_type = matched.get('instrument_type', None)
                    annotation.collision_energy = matched.get('collision_energy', None)
                    annotation.db_id = matched.get('comment', None)
                    annotation.matched_spec = matched.get('peaks', None)

                    spec.annotation_ls.append(annotation)

    return chunk


def refine_ms1_id_results(ms1_spec_ls, mz_tol=0.01, max_prec_rel_int=0.05):
    """
    Refine MS1 ID results within each pseudo MS2 spectrum using a cumulative public spectrum approach.

    :param ms1_spec_ls: List of PseudoMS2-like objects
    :param mz_tol: m/z tolerance for comparing precursor masses
    :param max_prec_rel_int: Maximum relative intensity threshold for precursor in public spectrum
    :return: Refined list of PseudoMS2-like objects
    """
    for spec in ms1_spec_ls:
        if spec.annotated and len(spec.annotation_ls) > 1:
            # Sort annotations by precursor m/z in descending order
            spec.annotation_ls.sort(key=lambda x: x.precursor_mz, reverse=True)

            public_mz = np.array([])  # Public spectrum, all matched peaks
            public_intensity = np.array([])
            to_keep = []

            for annotation in spec.annotation_ls:

                current_precursor_mz = annotation.precursor_mz

                # Check if precursor appears in public spectrum
                if public_mz.size > 0:
                    mz_diff = np.abs(public_mz - current_precursor_mz)
                    min_diff_idx = np.argmin(mz_diff)
                    if mz_diff[min_diff_idx] <= mz_tol and public_intensity[min_diff_idx] > max_prec_rel_int:
                        continue

                to_keep.append(annotation)

                # Add the reference spectrum to the public spectrum
                ref_spectrum = np.array(annotation.matched_spec)
                ref_mz = ref_spectrum[:, 0]
                ref_intensity = ref_spectrum[:, 1] / np.max(ref_spectrum[:, 1])

                if public_mz.size == 0:
                    public_mz = ref_mz
                    public_intensity = ref_intensity
                else:
                    # Add peaks
                    public_mz = np.concatenate([public_mz, ref_mz])
                    public_intensity = np.concatenate([public_intensity, ref_intensity])

            # Update annotations
            spec.annotation_ls = to_keep

    return ms1_spec_ls
