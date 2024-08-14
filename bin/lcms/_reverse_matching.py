import os
import pickle
from typing import List
import numpy as np
from ms_entropy import read_one_spectrum

from _utils import SpecAnnotation
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


def ms1_id_annotation(ms1_spec_ls, ms2_library, mz_tol=0.01,
                      score_cutoff=0.8, min_matched_peak=6,
                      ion_mode=None, refine=True,
                      max_prec_rel_int_in_other_ms2=0.05,
                      save=False, save_path=None):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :param ion_mode: str, ion mode, can be None (default), 'positive', or 'negative'
    :param refine: bool, refine the results
    :param max_prec_rel_int_in_other_ms2: float, maximum precursor relative intensity in other MS2 spectrum
    :param save: bool, save the results
    :param save_path: str, save directory
    :return: PseudoMS1-like object
    """

    # perform revcos matching
    ms1_spec_ls = ms1_id_revcos_matching(ms1_spec_ls, ms2_library, mz_tol=mz_tol,
                                         ion_mode=ion_mode,
                                         score_cutoff=score_cutoff,
                                         min_matched_peak=min_matched_peak)

    # refine the results, to avoid wrong annotations (ATP, ADP, AMP all annotated at the same RT)
    if refine:
        print('Refining MS1 ID results...')
        ms1_spec_ls = refine_ms1_id_results(ms1_spec_ls, mz_tol=mz_tol,
                                            max_prec_rel_int=max_prec_rel_int_in_other_ms2)

    if save and save_path is not None:
        with open(save_path, 'wb') as file:
            pickle.dump(ms1_spec_ls, file)

    return ms1_spec_ls


def ms1_id_revcos_matching(ms1_spec_ls: List, ms2_library: str, mz_tol: float = 0.02,
                           ion_mode: str = None, score_cutoff: float = 0.7,
                           min_matched_peak: int = 3):
    """
    Perform MS1 annotation using open search for the entire spectrum, with filters similar to identity search.

    :param ms1_spec_ls: a list of PseudoMS1-like objects
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: m/z tolerance in Da, for open matching
    :param ion_mode: str, ion mode, can be None (default), 'positive', or 'negative'
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :return: List of updated PseudoMS1-like objects
    """
    mz_tol = max(mz_tol, 0.02)  # indexed library mz_tol is 0.02

    # Load the data
    with open(ms2_library, 'rb') as file:
        search_eng = pickle.load(file)

    for spec in ms1_spec_ls:
        # Sort mzs and intensities in ascending order of mz
        sorted_indices = np.argsort(spec.mzs)
        sorted_mzs = np.array(spec.mzs)[sorted_indices]
        sorted_intensities = np.array(spec.intensities)[sorted_indices]

        matching_result = search_eng.search(
            precursor_mz=spec.t_mz,
            peaks=[[mz, intensity] for mz, intensity in zip(sorted_mzs, sorted_intensities)],
            ms1_tolerance_in_da=mz_tol,
            ms2_tolerance_in_da=mz_tol,
            method="open",
            precursor_ions_removal_da=0.5,
            noise_threshold=0.0,
            min_ms2_difference_in_da=mz_tol * 2.02,
            reverse=True
        )

        score_arr, matched_peak_arr, spec_usage_arr = matching_result['open_search']

        # filter by matching cutoffs, including scaled score
        v = np.where(np.logical_and(score_arr >= score_cutoff, matched_peak_arr >= min_matched_peak))[0]

        all_matches = []
        for idx in v:
            matched = {k.lower(): v for k, v in search_eng[idx].items()}

            this_ion_mode = matched.get('ion_mode', '')
            if ion_mode is not None and ion_mode != this_ion_mode:
                continue

            # precursor should be in the pseudo MS1 spectrum
            precursor_mz = matched.get('precursor_mz', 0)
            if not any(np.isclose(sorted_mzs, precursor_mz, atol=mz_tol)):
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

    return ms1_spec_ls


def refine_ms1_id_results(ms1_spec_ls, mz_tol=0.01, max_prec_rel_int=0.05):
    """
    Refine MS1 ID results within each pseudo MS1 spectrum using a NumPy-optimized cumulative public spectrum approach.

    :param ms1_spec_ls: List of PseudoMS1-like objects
    :param mz_tol: m/z tolerance for comparing precursor masses
    :param max_prec_rel_int: Maximum relative intensity threshold for precursor in public spectrum
    :return: Refined list of PseudoMS1-like objects
    """
    for spec in ms1_spec_ls:
        if spec.annotated and len(spec.annotation_ls) > 1:
            # Sort annotations by score in descending order
            spec.annotation_ls.sort(key=lambda x: x.score, reverse=True)

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
                max_intensity = np.max(ref_spectrum[:, 1])
                ref_mz = ref_spectrum[:, 0]
                ref_intensity = ref_spectrum[:, 1] / max_intensity

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

'''
def refine_ms1_id_results_for_identity_search(ms1_spec_ls, rt_tol=0.1, mz_tol=0.01, max_prec_rel_int=0.05):
    """
    Refine MS1 ID results across pseudo MS1 spectra.

    :param ms1_spec_ls: List of PseudoMS1-like objects
    :param rt_tol: Retention time tolerance for considering nearby spectra
    :param mz_tol: m/z tolerance for comparing precursor masses
    :param max_prec_rel_int: Maximum relative intensity threshold for precursor in reference spectra
    :return: Refined list of PseudoMS1-like objects
    """
    for i, spec in enumerate(ms1_spec_ls):
        # Check if the spectrum has annotations and all have the same InChIKey
        if spec.annotated and spec.annotation_ls:
            inchikeys = {ann.inchikey for ann in spec.annotation_ls if ann.inchikey}
            if len(inchikeys) == 1:
                unique_inchikey = next(iter(inchikeys))
                unique_rt = spec.rt

                # Use only the first reference spectrum in the annotation_ls
                unique_ref_spectrum = spec.annotation_ls[0].peaks
                if unique_ref_spectrum is None:
                    continue  # Skip if there's no reference spectrum

                max_intensity = max(peak[1] for peak in unique_ref_spectrum)

                # Find nearby spectra
                for j, nearby_spec in enumerate(ms1_spec_ls):
                    if i != j and abs(nearby_spec.rt - unique_rt) <= rt_tol:
                        to_remove = []
                        for k, annotation in enumerate(nearby_spec.annotation_ls):
                            # Check if the precursor of this annotation appears in the unique reference spectrum
                            precursor_intensity = next(
                                (peak[1] for peak in unique_ref_spectrum if
                                 abs(peak[0] - annotation.precursor_mz) <= mz_tol),
                                0
                            )
                            if precursor_intensity / max_intensity > max_prec_rel_int:
                                to_remove.append(k)

                        # Remove flagged annotations
                        for index in sorted(to_remove, reverse=True):
                            del nearby_spec.annotation_ls[index]

                        # Update annotation status
                        nearby_spec.annotated = len(nearby_spec.annotation_ls) > 0

    return ms1_spec_ls
'''

if __name__ == "__main__":
    # prepare_ms2_lib(ms2db='../../data/gnps.msp', mz_tol=0.02, peak_scale_k=None, peak_intensity_power=0.5)
    # prepare_ms2_lib(ms2db='../../data/nist20.msp', mz_tol=0.02, peak_scale_k=None, peak_intensity_power=0.5)
    # prepare_ms2_lib(ms2db='../../data/gnps_nist20.msp', mz_tol=0.02, peak_scale_k=None, peak_intensity_power=0.5)

    prepare_ms2_lib(ms2db='../../data/gnps.msp', mz_tol=0.02, peak_scale_k=10, peak_intensity_power=0.5)
    prepare_ms2_lib(ms2db='../../data/nist20.msp', mz_tol=0.02, peak_scale_k=10, peak_intensity_power=0.5)
    # prepare_ms2_lib(ms2db='../../data/gnps_nist20.msp', mz_tol=0.02, peak_scale_k=10, peak_intensity_power=0.5)

    # with open('../../data/gnps.pkl', 'rb') as file:
    #     search_eng = pickle.load(file)
    #
    # print(search_eng)
