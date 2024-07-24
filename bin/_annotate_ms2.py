import pickle
import re

import numpy as np


def feature_annotation(features, parameters, ms2id_score_cutoff=0.8, ms2id_min_matched_peak=6, num=5):
    """
    A function to annotate features based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    features : list
        A list of features.
    parameters : Params object
        The parameters for the workflow.
    num : int
        The number of top MS/MS spectra to search.
    """

    # load the MS/MS database
    search_eng = pickle.load(open(parameters.msms_library, 'rb'))

    ms2_tol = max(parameters.mz_tol_ms2, 0.02)  # indexed library mz_tol is 0.02

    for f in features:
        if len(f.ms2_seq) == 0:
            continue
        parsed_ms2 = []
        for ms2 in f.ms2_seq:
            peaks = _extract_peaks_from_string(ms2)
            parsed_ms2.append(peaks)
        # sort parsed ms2 by summed intensity
        parsed_ms2 = sorted(parsed_ms2, key=lambda x: np.sum(x[:, 1]), reverse=True)
        parsed_ms2 = parsed_ms2[:num]
        matched = None
        highest_similarity = 0
        matched_peak = 0
        f.best_ms2 = _convert_peaks_to_string(parsed_ms2[0])
        best_ms2 = parsed_ms2[0]
        for peaks in parsed_ms2:

            cos_result = search_eng.search(
                precursor_mz=f.mz,
                peaks=peaks,
                ms1_tolerance_in_da=parameters.mz_tol_ms1,
                ms2_tolerance_in_da=ms2_tol,
                method="identity",
                precursor_ions_removal_da=0.5,
                noise_threshold=0.001,
                min_ms2_difference_in_da=ms2_tol * 2.02
            )
            score_arr, matched_peak_arr, spec_usage_arr = cos_result['identity_search']

            idx = np.where(matched_peak_arr >= ms2id_min_matched_peak)[0]
            if len(idx) > 0:
                idx = idx[np.argmax(score_arr[idx])]
                if score_arr[idx] > ms2id_score_cutoff:
                    matched = search_eng[np.argmax(score_arr)]
                    matched = {k.lower(): v for k, v in matched.items()}
                    best_ms2 = peaks
                    highest_similarity = score_arr[idx]
                    matched_peak = matched_peak_arr[idx]

        if matched:
            f.annotation = matched.get('name', None)
            f.search_mode = 'identity_search'
            f.similarity = highest_similarity
            f.matched_peak_number = matched_peak
            f.smiles = matched.get('smiles', None)
            f.inchikey = matched.get('inchikey', None)
            f.matched_ms2 = _convert_peaks_to_string(matched['peaks'])
            f.formula = matched.get('formula', None)

            precursor_type = matched.get('precursor_type')
            if not precursor_type:
                precursor_type = matched.get('precursortype')
            f.adduct_type = precursor_type

            f.best_ms2 = _convert_peaks_to_string(best_ms2)

    return features


def annotate_rois(d, ms2id_score_cutoff=0.8, ms2id_min_matched_peak=6):
    """
    A function to annotate rois based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    d : MSData object
        MS data.
    """

    # load the MS/MS database
    search_eng = pickle.load(open(d.params.msms_library, 'rb'))

    ms2_tol = max(d.params.mz_tol_ms2, 0.02)  # indexed library mz_tol is 0.02

    for f in d.rois:
        f.annotation = None
        f.similarity = None
        f.matched_peak_number = None
        f.smiles = None
        f.inchikey = None
        f.matched_precursor_mz = None
        f.matched_peaks = None
        f.formula = None

        if f.best_ms2 is not None:

            cos_result = search_eng.search(
                precursor_mz=f.mz,
                peaks=f.best_ms2.peaks,
                ms1_tolerance_in_da=d.params.mz_tol_ms1,
                ms2_tolerance_in_da=ms2_tol,
                method="identity",
                precursor_ions_removal_da=None,
                noise_threshold=0.0,
                min_ms2_difference_in_da=ms2_tol * 2.2,
                max_peak_num=None
            )
            score_arr, matched_peak_arr, spec_usage_arr = cos_result['identity_search']

            idx = np.where(matched_peak_arr >= ms2id_min_matched_peak)[0]
            if len(idx) > 0:
                idx = idx[np.argmax(score_arr[idx])]
                if score_arr[idx] > ms2id_score_cutoff:
                    matched = search_eng[np.argmax(score_arr)]
                    matched = {k.lower(): v for k, v in matched.items()}
                    f.annotation = matched.get('name', None)
                    f.similarity = score_arr[idx]
                    f.matched_peak_number = matched_peak_arr[idx]
                    f.smiles = matched.get('smiles', None)
                    f.inchikey = matched.get('inchikey', None)
                    f.matched_precursor_mz = matched.get('precursor_mz', None)
                    f.matched_peaks = matched.get('peaks', None)
                    f.formula = matched.get('formula', None)


def _extract_peaks_from_string(ms2):
    """
    Extract peaks from MS2 spectrum.

    Parameters
    ----------
    ms2 : str
        MS2 spectrum in string format.

    Example
    ----------

    """

    # Use findall function to extract all numbers matching the pattern
    numbers = re.findall(r'\d+\.\d+', ms2)

    # Convert the extracted numbers from strings to floats
    numbers = [float(num) for num in numbers]

    numbers = np.array(numbers).reshape(-1, 2)

    return numbers


def _convert_peaks_to_string(peaks):
    """
    Convert peaks to string format.

    Parameters
    ----------
    peaks : numpy.array
        Peaks in numpy array format.

    Example
    ----------

    """

    ms2 = ""
    for i in range(len(peaks)):
        ms2 += str(np.round(peaks[i, 0], decimals=4)) + ";" + str(np.round(peaks[i, 1], decimals=4)) + "|"
    ms2 = ms2[:-1]

    return ms2
