import os
import pickle
from typing import List
import numpy as np
import pandas as pd
from ms_entropy import read_one_spectrum

from _utils import SpecAnnotation
from flash_cos import FlashCos


def prepare_ms2_lib(ms2db, mz_tol=0.02, sqrt_transform=True):
    """
    prepare ms2 db using MSP formatted database
    :return: a pickle file
    """

    db = []
    for a in read_one_spectrum(ms2db):
        db.append(a)

    print('Number of spectra in the database:', len(db))

    if 'precursor_mz' not in db[0].keys():
        for a in db:
            similar_key = [k for k in a.keys() if 'prec' in k and 'mz' in k]
            try:
                a['precursor_mz'] = float(a.pop(similar_key[0]))
            except:
                a['precursor_mz'] = 0.0
    else:
        for a in db:
            try:
                a['precursor_mz'] = float(a['precursor_mz'])
            except:
                a['precursor_mz'] = 0.0

    print('initializing search engine')
    search_engine = FlashCos(max_ms2_tolerance_in_da=mz_tol * 1.005,
                             mz_index_step=0.0001,
                             path_data=None,
                             sqrt_transform=sqrt_transform)
    print('building index')
    search_engine.build_index(db,
                              max_indexed_mz=2000,
                              precursor_ions_removal_da=0.5,
                              noise_threshold=0.001,
                              min_ms2_difference_in_da=mz_tol * 2.02,
                              clean_spectra=True)

    new_path = os.path.splitext(ms2db)[0] + '.pkl'
    # save as pickle
    with open(new_path, 'wb') as file:
        # Dump the data into the file
        pickle.dump(search_engine, file)

    print(f"Pickle file saved to: {new_path}")
    return search_engine


def ms1_id_annotation(ms1_spec_ls, ms2_library, mz_tol=0.01,
                      score_cutoff=0.8, min_matched_peak=6,
                      min_prec_int_in_ms1=1000,
                      max_prec_rel_int_in_other_ms2=0.05):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param min_prec_int_in_ms1: minimum required precursor intensity in MS1 spectrum
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :param max_prec_rel_int_in_other_ms2: float, maximum precursor relative intensity in other MS2 spectrum
    :return: PseudoMS1-like object
    """

    # perform revcos matching
    ms1_spec_ls = ms1_id_revcos_matching_new(ms1_spec_ls, ms2_library, mz_tol=mz_tol,
                                             min_prec_int_in_ms1=min_prec_int_in_ms1,
                                             score_cutoff=score_cutoff, min_matched_peak=min_matched_peak)

    # refine the results, to avoid wrong annotations (ATP, ADP, AMP all annotated at the same RT)
    ms1_spec_ls = refine_ms1_id_results(ms1_spec_ls, mz_tol=mz_tol,
                                        max_prec_rel_int_in_other_ms2=max_prec_rel_int_in_other_ms2)

    return ms1_spec_ls


def ms1_id_revcos_matching_new(ms1_spec_ls: List, ms2_library: str, mz_tol: float = 0.02,
                               min_prec_int_in_ms1: float = 1000, score_cutoff: float = 0.8,
                               min_matched_peak: int = 6) -> List:
    """
    Perform MS1 annotation using identity search for each precursor m/z.

    :param ms1_spec_ls: a list of PseudoMS1-like objects
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: m/z tolerance in Da, for identity matching
    :param min_prec_int_in_ms1: minimum precursor intensity in MS1 spectrum
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :return: List of updated PseudoMS1-like objects
    """
    mz_tol = max(mz_tol, 0.02)  # indexed library mz_tol is 0.02

    # Load the data
    with open(ms2_library, 'rb') as file:
        search_eng = pickle.load(file)

    for spec in ms1_spec_ls:
        if len(spec.mzs) < min_matched_peak:
            spec.annotated = False
            continue

        # Sort mzs and intensities in ascending order of mz
        sorted_indices = np.argsort(spec.mzs)
        sorted_mzs = np.array(spec.mzs)[sorted_indices]
        sorted_intensities = np.array(spec.intensities)[sorted_indices]

        all_matches = []
        for prec_mz in sorted_mzs[min_matched_peak - 1:]:

            matching_result = search_eng.search(
                precursor_mz=prec_mz,
                peaks=[[mz, intensity] for mz, intensity in zip(sorted_mzs, sorted_intensities)],
                ms1_tolerance_in_da=mz_tol,
                ms2_tolerance_in_da=mz_tol,
                method="identity",
                precursor_ions_removal_da=0.5,  # reserve mzs up to prec_mz + mz_tol
                noise_threshold=0.001,
                min_ms2_difference_in_da=mz_tol * 2.02
            )

            score_arr, matched_peak_arr, spec_usage_arr = matching_result['identity_search']

            # filter by matching cutoffs
            v = np.where(np.logical_and(score_arr >= score_cutoff, matched_peak_arr >= min_matched_peak))[0]

            for idx in v:
                all_matches.append((idx, score_arr[idx], matched_peak_arr[idx], spec_usage_arr[idx], prec_mz))

        # Filter matches based on precursor relative intensity in MS1
        valid_matches = []
        for idx, score, matched_peaks, spectral_usage, prec_mz in all_matches:
            matched = {k.lower(): v for k, v in search_eng[idx].items()}
            precursor_mz = matched.get('precursor_mz')

            if precursor_mz is not None:
                closest_mz_idx = np.argmin(np.abs(sorted_mzs - precursor_mz))
                if abs(sorted_mzs[closest_mz_idx] - precursor_mz) <= mz_tol:
                    prec_intensity = sorted_intensities[closest_mz_idx]
                    if prec_intensity >= min_prec_int_in_ms1:
                        valid_matches.append((idx, score, matched_peaks, spectral_usage, prec_mz))

        if valid_matches:
            spec.annotated = True
            spec.annotation_ls = []
            for idx, score, matched_peaks, spectral_usage, prec_mz in valid_matches:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}

                annotation = SpecAnnotation(idx, score, matched_peaks)
                annotation.spectral_usage = spectral_usage
                annotation.intensity = sorted_intensities[np.argmin(np.abs(sorted_mzs - prec_mz))]

                annotation.name = matched.get('name', None)
                annotation.precursor_mz = matched.get('precursor_mz')

                precursor_type = matched.get('precursor_type', None)
                if not precursor_type:
                    precursor_type = matched.get('precursortype', None)
                annotation.precursor_type = precursor_type

                annotation.formula = matched.get('formula', None)
                annotation.inchikey = matched.get('inchikey', None)
                annotation.instrument_type = matched.get('instrument_type', None)
                annotation.collision_energy = matched.get('collision_energy', None)
                annotation.peaks = matched.get('peaks', None)

                spec.annotation_ls.append(annotation)

            # sort the annotations by precursor m/z in descending order
            spec.annotation_ls = sorted(spec.annotation_ls, key=lambda x: x.precursor_mz, reverse=True)
        else:
            spec.annotated = False

    return ms1_spec_ls


def refine_ms1_id_results(ms1_spec_ls, mz_tol=0.01, max_prec_rel_int_in_other_ms2=0.05):
    """
    Refine the ms1 id results, to avoid redundant annotations (ATP, ADP, AMP all annotated at the same RT)
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param mz_tol: float, mz tolerance
    :param max_prec_rel_int_in_other_ms2: float, maximum precursor relative intensity in other MS2 spectrum
    :return: refined ms1_spec_ls
    """

    for spec in ms1_spec_ls:
        if not spec.annotated or len(spec.annotation_ls) == 0:
            continue

        all_precursor_mzs = np.array([annotation.precursor_mz for annotation in spec.annotation_ls])
        annotation_bool_arr = np.array([True] * len(spec.annotation_ls))

        for i, annotation in enumerate(spec.annotation_ls):
            # Skip if this annotation is already labeled as False
            if not annotation_bool_arr[i]:
                continue

            # Get the peaks from the annotation
            peaks = annotation.peaks
            if peaks is None or len(peaks) == 0:
                continue

            # Get the m/z values of the peaks with relative intensities >= max_prec_rel_int_in_other_ms2
            max_intensity = np.max(peaks[:, 1])
            peak_mzs = peaks[:, 0][peaks[:, 1] / max_intensity >= max_prec_rel_int_in_other_ms2]

            # Check which precursor mzs exist in the peak mz list
            matches = np.any(np.abs(all_precursor_mzs[i + 1:, np.newaxis] - peak_mzs) <= mz_tol, axis=1)
            annotation_bool_arr[i + 1:][matches] = False

        # Remove annotations that didn't pass the checks
        spec.annotation_ls = [ann for ann, keep in zip(spec.annotation_ls, annotation_bool_arr) if keep]

    return ms1_spec_ls


if __name__ == "__main__":
    # prepare_ms2_lib(ms2db='../data/ALL_GNPS_NO_PROPOGATED.msp', mz_tol=0.02, sqrt_transform=True)

    with open('../data/ALL_GNPS_NO_PROPOGATED.pkl', 'rb') as file:
        search_eng = pickle.load(file)

    print(search_eng)
