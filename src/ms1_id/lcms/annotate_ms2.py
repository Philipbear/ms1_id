import pickle
import re

import numpy as np

from ms1_id.lcms.centroid_data import centroid_spectrum_for_search


def feature_annotation(features, config, num=5):
    """
    A function to annotate features based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    features : list
        A list of features.
    config : Params object
        The parameters for the workflow.
    num : int
        The number of top MS/MS spectra to search.
    """

    # load the MS/MS database
    search_eng = pickle.load(open(config.ms2id_library_path, 'rb'))

    library_search_mztol = min(config.library_search_mztol, 0.05)  # indexed library mz_tol is 0.05

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

        all_valid_matches = []

        for peaks in parsed_ms2:
            peaks = centroid_spectrum_for_search(peaks, width_da=0.05 * 2.015)

            if len(peaks) < config.ms2id_min_matched_peak:
                continue

            search_result = search_eng.search(
                precursor_mz=f.mz,
                peaks=peaks,
                ms1_tolerance_in_da=library_search_mztol,
                ms2_tolerance_in_da=library_search_mztol,
                method="identity",
                precursor_ions_removal_da=0.5,
                noise_threshold=0.0,
                min_ms2_difference_in_da=library_search_mztol * 2.02,
                reverse=False
            )
            score_arr, matched_peak_arr, spec_usage_arr = search_result['identity_search']

            # Find indices that pass all filters
            valid_indices = np.where((matched_peak_arr >= config.ms2id_min_matched_peak) &
                                     (score_arr >= config.ms2id_score_cutoff))[0]

            for idx in valid_indices:
                potential_match = search_eng[idx]
                potential_match = {k.lower(): v for k, v in potential_match.items()}

                if potential_match.get('ion_mode', '') == config.ion_mode:
                    all_valid_matches.append({
                        'match': potential_match,
                        'score': score_arr[idx],
                        'matched_peak': matched_peak_arr[idx],
                        'ms2': peaks
                    })

        # Select the match with the highest score
        if all_valid_matches:
            best_match = max(all_valid_matches, key=lambda x: x['score'])

            f.annotation = best_match['match'].get('name', None)
            f.search_mode = 'identity_search'
            f.similarity = best_match['score']
            f.matched_peak_number = best_match['matched_peak']
            f.smiles = best_match['match'].get('smiles', None)
            f.inchikey = best_match['match'].get('inchikey', None)
            f.matched_ms2 = _convert_peaks_to_string(best_match['match']['peaks'])
            f.formula = best_match['match'].get('formula', None)
            f.adduct_type = best_match['match'].get('precursor_type')
            f.best_ms2 = _convert_peaks_to_string(best_match['ms2'])
            f.collision_energy = best_match['match'].get('collision_energy', None)
            f.precursor_type = best_match['match'].get('precursor_type', None)
        else:
            f.best_ms2 = _convert_peaks_to_string(parsed_ms2[0])

    return features


def annotate_rois(d, ms2id_score_cutoff=0.8, ms2id_min_matched_peak=6, ion_mode='positive'):
    """
    A function to annotate rois based on their MS/MS spectra and a MS/MS database.

    Parameters
    ----------
    d : MSData object
        MS data.
    """

    # load the MS/MS database
    search_eng = pickle.load(open(d.params.ms2id_library_path, 'rb'))

    ms2_tol = min(d.params.mz_tol_ms2, 0.05)  # indexed library mz_tol is 0.05

    for f in d.rois:
        f.annotation = None
        f.similarity = None
        f.matched_peak_number = None
        f.smiles = None
        f.inchikey = None
        f.matched_precursor_mz = None
        f.matched_peaks = None
        f.formula = None
        f.precursor_type = None
        f.collision_energy = None

        if f.best_ms2 is not None:

            peaks = f.best_ms2.peaks

            # centroid
            peaks = centroid_spectrum_for_search(peaks, width_da=0.05 * 2.015)

            if len(peaks) < ms2id_min_matched_peak:
                continue

            search_result = search_eng.search(
                precursor_mz=f.mz,
                peaks=peaks,
                ms1_tolerance_in_da=d.params.mz_tol_ms1,
                ms2_tolerance_in_da=ms2_tol,
                method="identity",
                precursor_ions_removal_da=0.5,
                noise_threshold=0.0,
                min_ms2_difference_in_da=ms2_tol * 2.2,
                reverse=False
            )
            score_arr, matched_peak_arr, spec_usage_arr = search_result['identity_search']

            # Find indices that pass all filters
            valid_indices = np.where((matched_peak_arr >= ms2id_min_matched_peak) &
                                     (score_arr >= ms2id_score_cutoff))[0]

            all_valid_matches = []

            for idx in valid_indices:
                potential_match = search_eng[idx]
                potential_match = {k.lower(): v for k, v in potential_match.items()}

                if potential_match.get('ion_mode', '') == ion_mode:
                    all_valid_matches.append({
                        'match': potential_match,
                        'score': score_arr[idx],
                        'matched_peak': matched_peak_arr[idx]
                    })

            # Select the match with the highest score
            if all_valid_matches:
                best_match = max(all_valid_matches, key=lambda x: x['score'])

                f.annotation = best_match['match'].get('name', None)
                f.similarity = best_match['score']
                f.matched_peak_number = best_match['matched_peak']
                f.smiles = best_match['match'].get('smiles', None)
                f.inchikey = best_match['match'].get('inchikey', None)
                f.matched_precursor_mz = best_match['match'].get('precursor_mz', None)
                f.matched_peaks = best_match['match'].get('peaks', None)
                f.formula = best_match['match'].get('formula', None)
                f.precursor_type = best_match['match'].get('precursor_type', None)
                f.collision_energy = best_match['match'].get('collision_energy', None)


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
