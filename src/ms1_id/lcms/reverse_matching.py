import os
import pickle
import numpy as np
from ms_entropy import read_one_spectrum

from ms1_id.lcms.utils import SpecAnnotation
from ms1_id.search_utils.flash_cos import FlashCos
from ms1_id.lcms.centroid_data import centroid_spectrum_for_search


def prepare_ms2_lib(ms2db,
                    mz_tol: float = 0.05,
                    peak_scale_k: float = None,
                    peak_intensity_power: float = 0.5):
    """
    prepare ms2 db using MSP or MGF formatted database
    :param ms2db: path to the MSP formatted database
    :param mz_tol: mz tolerance
    :param peak_scale_k: peak scaling factor, None for no scaling
    :param peak_intensity_power: peak intensity power, 0.5 for square root
    :return: a pickle file
    """

    replace_keys = {'precursormz': 'precursor_mz',
                    'pepmass': 'precursor_mz',
                    'precursortype': 'precursor_type',
                    'ionmode': 'ion_mode',
                    'instrumenttype': 'instrument_type',
                    'collisionenergy': 'collision_energy',
                    'spectrumid': 'comment'}

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
            ion_mode = a['ion_mode'].lower().strip()
            if ion_mode in ['p', 'positive']:
                a['ion_mode'] = 'positive'
            elif ion_mode in ['n', 'negative']:
                a['ion_mode'] = 'negative'
            else:
                a['ion_mode'] = 'unknown'

        db.append(a)

    print('Number of spectra in the database:', len(db))

    print('Initializing search engine')
    search_engine = FlashCos(max_ms2_tolerance_in_da=mz_tol * 1.005,
                             mz_index_step=0.0001,
                             peak_scale_k=peak_scale_k,
                             peak_intensity_power=peak_intensity_power)
    print('Building index')
    search_engine.build_index(db,
                              max_indexed_mz=2000,
                              precursor_ions_removal_da=0.5,
                              noise_threshold=0.01,
                              min_ms2_difference_in_da=mz_tol * 2.02,
                              clean_spectra=True)

    if peak_scale_k is None:
        new_path = os.path.splitext(ms2db)[0] + '.pkl'
    else:
        new_path = os.path.splitext(ms2db)[0] + f'_k{peak_scale_k}.pkl'

    # save as pickle
    with open(new_path, 'wb') as file:
        pickle.dump(search_engine, file)

    print(f"Pickle file saved to: {new_path}")
    return search_engine


def ms1_id_annotation(ms1_spec_ls, library_ls, mz_tol=0.05,
                      score_cutoff=0.8, min_matched_peak=6, min_spec_usage=0.05,
                      ion_mode=None, refine=True,
                      max_prec_rel_int_in_other_ms2=0.05,
                      save=False, save_path=None):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param library_ls: list of path to the pickle file, indexed library
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :param min_spec_usage: for rev cos
    :param ion_mode: str, ion mode, can be None (default), 'positive', or 'negative'
    :param refine: bool, refine the results
    :param max_prec_rel_int_in_other_ms2: float, maximum precursor relative intensity in other MS2 spectrum
    :param save: bool, save the results
    :param save_path: str, save directory
    :return: PseudoMS2-like object
    """

    # perform revcos matching
    ms1_spec_ls = ms1_id_revcos_matching(ms1_spec_ls, library_ls, mz_tol=mz_tol,
                                         ion_mode=ion_mode,
                                         score_cutoff=score_cutoff,
                                         min_matched_peak=min_matched_peak,
                                         min_spec_usage=min_spec_usage)

    # refine the results, to avoid wrong annotations (ATP, ADP, AMP all annotated at the same RT)
    if refine:
        print('Refining MS1 ID results...')
        ms1_spec_ls = refine_ms1_id_results(ms1_spec_ls, mz_tol=mz_tol,
                                            max_prec_rel_int=max_prec_rel_int_in_other_ms2)

    if save and save_path is not None:
        with open(save_path, 'wb') as file:
            pickle.dump(ms1_spec_ls, file)

    return ms1_spec_ls


def ms1_id_revcos_matching(ms1_spec_ls, library_ls, mz_tol=0.02,
                           ion_mode=None, score_cutoff=0.7,
                           min_matched_peak=3, min_spec_usage=0.0):
    """
    Perform MS1 annotation using open search for the entire spectrum, with filters similar to identity search.

    :param ms1_spec_ls: a list of PseudoMS2-like objects
    :param library: path to the pickle file, indexed library
    :param mz_tol: m/z tolerance in Da, for open matching
    :param ion_mode: str, ion mode, can be None (default), 'positive', or 'negative'
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :param min_spec_usage: minimum spectral usage
    :return: List of updated PseudoMS2-like objects
    """
    mz_tol = min(mz_tol, 0.05)  # indexed library mz_tol is 0.05

    # centroid peaks
    for spec in ms1_spec_ls:
        peaks = list(zip(spec.mzs, spec.intensities))
        peaks = np.asarray(peaks, dtype=np.float32, order="C")
        peaks = centroid_spectrum_for_search(peaks, width_da=0.05 * 2.015)
        spec.centroided_peaks = peaks

    # if library_ls is a string, convert to list
    if isinstance(library_ls, str):
        library_ls = [library_ls]

    # Load the data
    for library in library_ls:
        with open(library, 'rb') as file:
            search_eng = pickle.load(file)
        db_name = os.path.basename(library)

        for spec in ms1_spec_ls:

            if len(spec.centroided_peaks) < min_matched_peak:
                continue

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
            for idx in v:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}

                this_ion_mode = matched.get('ion_mode', '')
                if ion_mode is not None and ion_mode != this_ion_mode:
                    continue

                # precursor should be in the pseudo MS2 spectrum
                precursor_mz = matched.get('precursor_mz', 0)
                if not any(np.isclose(np.array(spec.mzs), precursor_mz, atol=mz_tol)):
                    continue

                all_matches.append((idx, score_arr[idx], matched_peak_arr[idx], spec_usage_arr[idx]))

            if all_matches:
                spec.annotated = True
                for idx, score, matched_peaks, spectral_usage in all_matches:
                    matched = {k.lower(): v for k, v in search_eng[idx].items()}

                    annotation = SpecAnnotation(spec.mzs, spec.intensities, matched.get('precursor_mz'),
                                                db_name, idx, score, matched_peaks, spectral_usage, mz_tol)

                    annotation.name = matched.get('name', '')
                    annotation.precursor_type = matched.get('precursor_type', None)
                    annotation.formula = matched.get('formula', None)
                    annotation.inchikey = matched.get('inchikey', None)
                    annotation.instrument_type = matched.get('instrument_type', None)
                    annotation.collision_energy = matched.get('collision_energy', None)
                    annotation.db_id = matched.get('comment', None)
                    annotation.matched_spec = matched.get('peaks', None)

                    spec.annotation_ls.append(annotation)

    return ms1_spec_ls


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


if __name__ == "__main__":
    ######### prepare the search engine #########
    prepare_ms2_lib(ms2db='/Users/shipei/Documents/projects/ms1_id/data/gnps.msp',
                    mz_tol=0.05, peak_scale_k=None, peak_intensity_power=0.5)
    prepare_ms2_lib(ms2db='/Users/shipei/Documents/projects/ms1_id/data/gnps.msp',
                    mz_tol=0.05, peak_scale_k=10, peak_intensity_power=0.5)

    ######### load the search engine #########
    # with open('/Users/shipei/Documents/projects/ms1_id/data/gnps.pkl', 'rb') as file:
    #     search_eng = pickle.load(file)
    #
    # print(search_eng)
