import os
import pickle

import numpy as np
import pandas as pd
from ms_entropy import read_one_spectrum

from _utils import SpecAnnotation
from flash_revcos_search import FlashRevcosSearch


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
    search_engine = FlashRevcosSearch(max_ms2_tolerance_in_da=mz_tol * 1.05,
                                      mz_index_step=0.0001,
                                      low_memory=False,
                                      path_data=None,
                                      sqrt_transform=sqrt_transform)
    print('building index')
    search_engine.build_index(db,
                              max_indexed_mz=1500.0,
                              precursor_ions_removal_da=None,
                              noise_threshold=0.0,
                              min_ms2_difference_in_da=mz_tol * 2.2,
                              max_peak_num=-1,
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
    ms1_spec_ls = ms1_id_revcos_matching(ms1_spec_ls, ms2_library, mz_tol=mz_tol,
                                         min_prec_int_in_ms1=min_prec_int_in_ms1,
                                         score_cutoff=score_cutoff, min_matched_peak=min_matched_peak)

    # refine the results, to avoid wrong annotations (ATP, ADP, AMP all annotated at the same RT)
    ms1_spec_ls = refine_ms1_id_results(ms1_spec_ls, mz_tol=mz_tol,
                                        max_prec_rel_int_in_other_ms2=max_prec_rel_int_in_other_ms2)

    return ms1_spec_ls


def ms1_id_revcos_matching(ms1_spec_ls, ms2_library, mz_tol=0.02, min_prec_int_in_ms1=1000,
                           score_cutoff=0.8, min_matched_peak=6):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param min_prec_int_in_ms1: minimum precursor intensity in MS1 spectrum
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :return: PseudoMS1-like object
    """

    mz_tol = max(mz_tol, 0.02)  # indexed library mz_tol is 0.02

    # Load the data
    with open(ms2_library, 'rb') as file:
        search_eng = pickle.load(file)

    for spec in ms1_spec_ls:
        if len(spec.mzs) == 0:
            continue

        matching_result = search_eng.search(
            precursor_mz=0.0,
            peaks=[[mz, intensity] for mz, intensity in zip(spec.mzs, spec.intensities)],
            ms1_tolerance_in_da=0.02,
            ms2_tolerance_in_da=mz_tol,
            method="open",
            precursor_ions_removal_da=1.6,
            noise_threshold=0.0,
            min_ms2_difference_in_da=mz_tol * 2.2,
            max_peak_num=None
        )

        score_arr, matched_peak_arr, spec_usage_arr = matching_result['open_search']

        # filter by matching cutoffs
        v = np.where(np.logical_and(score_arr >= score_cutoff, matched_peak_arr >= min_matched_peak))[0]

        if len(v) > 0:
            # Filter matches based on precursor relative intensity in MS1
            valid_matches = []
            max_intensity = max(spec.intensities)
            for idx in v:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}
                precursor_mz = matched.get('precursor_mz')

                # ensure precursor_mz min_prec_rel_int_in_ms1 in the MS1 spectrum
                if precursor_mz is not None:
                    # Find the closest m/z in the spectrum to the precursor m/z
                    closest_mz_idx = np.argmin(np.abs(np.array(spec.mzs) - precursor_mz))
                    if abs(spec.mzs[closest_mz_idx] - precursor_mz) <= mz_tol:
                        prec_intensity = spec.intensities[closest_mz_idx]
                        if prec_intensity >= min_prec_int_in_ms1:
                            valid_matches.append(idx)
            v = np.array(valid_matches)

            if len(v) > 0:
                for matched_index in v:
                    # Get the details of each match
                    matched_score = score_arr[matched_index]
                    matched_peaks = matched_peak_arr[matched_index]
                    spectral_usage = spec_usage_arr[matched_index]

                    # Get the corresponding spectrum from the search engine's database
                    matched_spectrum = search_eng[matched_index]
                    matched = {k.lower(): q for k, q in matched_spectrum.items()}

                    # Create a SpecAnnotation object for the match
                    annotation = SpecAnnotation(matched_index, matched_score, matched_peaks)

                    annotation.spectral_usage = spectral_usage

                    # annotation intensity: the intensity of the precursor peak in the MS1 spectrum
                    annotation.intensity = spec.intensities[
                        np.argmin(np.abs(np.array(spec.mzs) - matched.get('precursor_mz')))]

                    # Fill in the database info from the matched spectrum
                    annotation.name = matched.get('name')
                    annotation.precursor_mz = matched.get('precursor_mz')
                    annotation.precursor_type = matched.get('precursor_type')
                    annotation.formula = matched.get('formula')
                    annotation.inchikey = matched.get('inchikey')
                    annotation.instrument_type = matched.get('instrument_type')
                    annotation.collision_energy = matched.get('collision_energy')
                    annotation.peaks = matched.get('peaks')

                    # Add the annotation to the spectrum
                    spec.annotated = True
                    spec.annotation_ls.append(annotation)

                # sort the annotations by precursor m/z in descending order
                spec.annotation_ls = sorted(spec.annotation_ls, key=lambda x: x.precursor_mz, reverse=True)

            else:
                spec.annotated = False
        else:
            spec.annotated = False

    # After the loop, ms1_spec_ls will have updated annotations
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
    prepare_ms2_lib(ms2db='../data/ALL_GNPS_NO_PROPOGATED.msp', mz_tol=0.02, sqrt_transform=True)
