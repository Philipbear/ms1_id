import os.path
import pickle
import numpy as np
from flash_revcos_search import FlashRevcosSearch
from ms_entropy import read_one_spectrum
from group_ppc_aligned_feature import PseudoMS1, SpecAnnotation


def prepare_ms2_lib(ms2db, save_path, mz_tol=0.01):
    """
    prepare ms2 db using MSP formatted database
    :return: a pickle file
    """

    db = []
    for a in read_one_spectrum(save_path):
        db.append(a)

    if 'precursor_mz' not in db[0].keys():
        for a in db:
            similar_key = [k for k in a.keys() if 'prec' in k and 'mz' in k]
            a['precursor_mz'] = float(a.pop(similar_key[0]))

    search_engine = FlashRevcosSearch(max_ms2_tolerance_in_da=mz_tol * 1.05,
                                      mz_index_step=0.0001,
                                      low_memory=False,
                                      path_data=None,
                                      sqrt_transform=False)

    search_engine.build_index(db,
                              max_indexed_mz=1500.0,
                              precursor_ions_removal_da=None,
                              noise_threshold=0.0,
                              min_ms2_difference_in_da=mz_tol * 2.2,
                              max_peak_num=-1,
                              clean_spectra=True)

    new_path = os.path.join(save_path, ms2db.split('.')[0] + '.pkl')
    # save as pickle
    with open(new_path, 'wb') as file:
        # Dump the data into the file
        pickle.dump(search_engine, file)

    print(f"Pickle file saved to: {new_path}")


def ms1_id_annotation(ms1_spec_ls, ms2_library, mz_tol=0.01,
                      score_cutoff=0.8, min_matched_peak=6):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param score_cutoff: for rev cos
    :param min_matched_peak: for rev cos
    :return: PseudoMS1-like object
    """

    # Load the data
    with open(ms2_library, 'rb') as file:
        search_eng = pickle.load(file)

    for spec in ms1_spec_ls:
        if len(spec.mzs) == 0:
            continue

        revcos_result = search_eng.search(
            precursor_mz=0.0,
            peaks=[[mz, intensity] for mz, intensity in zip(spec.mzs, spec.intensities)],
            ms1_tolerance_in_da=0.02,
            ms2_tolerance_in_da=mz_tol,
            method="open",
            precursor_ions_removal_da=None,
            noise_threshold=0.0,
            min_ms2_difference_in_da=mz_tol * 2.2,
            max_peak_num=None
        )

        score_arr, matched_peak_arr, spec_usage_arr = revcos_result['open_search']


