import os
import pickle
import numpy as np
import pandas as pd
from flash_revcos_search import FlashRevcosSearch
from ms_entropy import read_one_spectrum
from group_ppc_aligned_feature import SpecAnnotation


def prepare_ms2_lib(ms2db, mz_tol=0.01):
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

    print('initializing search engine')
    search_engine = FlashRevcosSearch(max_ms2_tolerance_in_da=mz_tol * 1.05,
                                      mz_index_step=0.0001,
                                      low_memory=False,
                                      path_data=None,
                                      sqrt_transform=False)
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


def ms1_id_annotation(ms1_spec_ls, ms2_library, mz_tol=0.01, precursor_in_spec=True,
                      score_cutoff=0.8, min_matched_peak=6):
    """
    Perform ms1 annotation
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param ms2_library: path to the pickle file, indexed library
    :param mz_tol: mz tolerance in Da, for rev cos matching
    :param precursor_in_spec: whether to ask the target precursor exists in the spectrum
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

        matching_result = search_eng.search(
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

        score_arr, matched_peak_arr, spec_usage_arr = matching_result['open_search']

        # filter by matching cutoffs
        v = np.where(np.logical_and(score_arr >= score_cutoff, matched_peak_arr >= min_matched_peak))[0]

        if len(v) > 0:
            if precursor_in_spec:
                # Filter matches where precursor m/z is in the query spectrum
                valid_matches = []
                for idx in v:
                    matched = {k.lower(): v for k, v in search_eng[idx].items()}
                    precursor_mz = matched.get('precursor_mz')
                    if precursor_mz is not None:
                        # Check if precursor m/z is in the query spectrum (within mz_tol)
                        if any(abs(mz - precursor_mz) <= mz_tol for mz in spec.mzs):
                            valid_matches.append(idx)
                v = np.array(valid_matches)

            if len(v) > 0:
                # Find the index of the match with the highest score
                best_match_index = v[np.argmax(score_arr[v])]

                # Get the details of the best match
                best_score = score_arr[best_match_index]
                best_matched_peaks = matched_peak_arr[best_match_index]
                best_spec_usage = spec_usage_arr[best_match_index]

                # Get the corresponding spectrum from the search engine's database
                best_match_spectrum = search_eng[best_match_index]
                matched = {k.lower(): v for k, v in best_match_spectrum.items()}

                # Create a SpecAnnotation object for the best match
                annotation = SpecAnnotation(best_score, best_matched_peaks)

                # Fill in the database info from the best match spectrum
                annotation.db_id = matched.get('id')
                annotation.name = matched.get('name')
                annotation.precursor_type = matched.get('precursor_type')
                annotation.formula = matched.get('formula')
                annotation.inchikey = matched.get('inchikey')
                annotation.instrument_type = matched.get('instrument_type')
                annotation.collision_energy = matched.get('collision_energy')

                # Add the annotation to the spectrum
                spec.annotated = True
                spec.annotation_ls.append(annotation)
            else:
                spec.annotated = False
        else:
            spec.annotated = False

    # After the loop, ms1_spec_ls will have updated annotations
    return ms1_spec_ls


def write_ms1_id_results(ms1_spec_ls, out_path):
    """
    Output the annotated ms1 spectra
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param out_path: output path
    :return: None
    """

    # only write out spectra with annotations
    ms1_spec_ls = [spec for spec in ms1_spec_ls if spec.annotated]

    out_list = []
    for spec in ms1_spec_ls:
        for annotation in spec.annotation_ls:
            # pseudo_ms1_str: mz1 int1; mz2 int2; ...
            pseudo_ms1_str = ' '.join([f"{mz} {intensity}" for mz, intensity in zip(spec.mzs, spec.intensities)])
            out_list.append({
                'rt': spec.rt,
                'pseudo_ms1': pseudo_ms1_str,
                'score': round(annotation.score, 4),
                'matched_peak': annotation.matched_peak,
                'db_id': annotation.db_id,
                'name': annotation.name,
                'precursor_type': annotation.precursor_type,
                'formula': annotation.formula,
                'inchikey': annotation.inchikey,
                'instrument_type': annotation.instrument_type,
                'collision_energy': annotation.collision_energy
            })

    out_df = pd.DataFrame(out_list)
    out_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # prepare_ms2_lib(ms2db='/Users/shipei/Documents/projects/ms1_id/data/ALL_GNPS_NO_PROPOGATED.msp', mz_tol=0.01)
    prepare_ms2_lib(ms2db='/Users/shipei/Documents/projects/ms1_id/data/MassBank_NIST.msp', mz_tol=0.01)
