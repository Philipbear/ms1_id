"""
Annotate pseudo MS/MS spectra in MGF format using indexed MS/MS libraries.
"""

import os
import pickle

import numpy as np
import pandas as pd
from ms_entropy import read_one_spectrum
from tqdm import tqdm

from ms1_id.lcms.centroid_data import centroid_spectrum_for_search


def annotate_pseudo_ms2_spec(ms2_file_path, library_ls,
                             mz_tol=0.05,
                             ion_mode=None,
                             score_cutoff=0.7,
                             min_matched_peak=3,
                             min_spec_usage=0.20,
                             save_dir=None):
    """
    Annotate pseudo MS/MS spectra in MGF format using indexed MS/MS libraries.

    :param ms2_file_path: path to the MGF file containing pseudo MS2 spectra
    :param library_ls: list of paths to indexed MS/MS libraries
    :param mz_tol: m/z tolerance in Da, for open matching
    :param ion_mode: ion mode, 'positive' or 'negative' or None
    :param score_cutoff: minimum score for matching
    :param min_matched_peak: minimum number of matched peaks
    :param min_spec_usage: minimum spectral usage
    :param save_dir: path to save the annotated results
    :return: DataFrame containing the annotation results
    """
    mz_tol = min(mz_tol, 0.05)  # indexed library mz_tol is 0.05

    # if library_ls is a string, convert to list
    if isinstance(library_ls, str):
        library_ls = [library_ls]

    all_matches = []

    # Load db
    for library in library_ls:
        with open(library, 'rb') as file:
            search_eng = pickle.load(file)
        db_name = os.path.basename(library)
        print(f'Loaded {db_name}')

        spec_idx = 0
        for spec in tqdm(read_one_spectrum(ms2_file_path)):

            spec_idx += 1

            # qry spec id
            spec_id = spec.get('title', None) or spec.get('spectrum_id', None) or spec.get('spectrumid', None) or \
                      spec.get('feature_id', None) or spec.get('featureid', None) or spec.get('scans', None)
            if spec_id is None:
                spec_id = spec_idx

            # qry precursor mz
            spec_mz = spec.get('precursor_mz', None) or spec.get('precursormz', None) or spec.get('pepmass', None)

            # skip if no peaks
            if not spec.get('peaks', None):
                continue

            peaks = np.asarray(spec['peaks'], dtype=np.float32, order="C")
            peaks = centroid_spectrum_for_search(peaks, width_da=0.05 * 2.015)

            if len(peaks) < min_matched_peak:
                continue

            matching_result = search_eng.search(
                precursor_mz=2000.00,  # unused, open search
                peaks=peaks,
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

            if len(v) == 0:
                continue

            for idx in v:
                matched = {k.lower(): v for k, v in search_eng[idx].items()}

                this_ion_mode = matched.get('ion_mode', '')
                if ion_mode is not None and ion_mode != this_ion_mode:
                    continue

                # precursor should be in the pseudo MS2 spectrum
                precursor_mz = matched.get('precursor_mz', 0)
                if not any(np.isclose(peaks[:, 0], precursor_mz, atol=mz_tol)):
                    continue

                all_matches.append({
                    'id': spec_id,
                    'mz': spec_mz,
                    'score': score_arr[idx],
                    'matched_peaks': matched_peak_arr[idx],
                    'spectral_usage': spec_usage_arr[idx],
                    'ref_name': matched.get('name', None),
                    'ref_precursor_mz': matched.get('precursor_mz', None),
                    'ref_precursor_type': matched.get('precursor_type', None),
                    'ref_formula': matched.get('formula', None),
                    'ref_inchikey': matched.get('inchikey', None),
                    'ref_instrument_type': matched.get('instrument_type', None),
                    'ref_collision_energy': matched.get('collision_energy', None),
                    'ref_comment': matched.get('comment', None),
                    'ref_db_name': db_name
                })

    df = pd.DataFrame(all_matches)

    # sort by id, then score
    df_top1 = df.sort_values(by=['id', 'score'], ascending=[True, False]).reset_index(drop=True)
    df_top1 = df_top1.groupby('id').head(1)

    out_basename = os.path.splitext(os.path.basename(ms2_file_path))[0] + '_annotations_all.tsv'
    out_top1_basename = os.path.splitext(os.path.basename(ms2_file_path))[0] + '_annotations_top1.tsv'
    if save_dir is not None:
        out_dir = save_dir
    else:
        # folder of the input file
        out_dir = os.path.dirname(ms2_file_path)

    df.to_csv(os.path.join(out_dir, out_basename), sep='\t', index=False)
    df_top1.to_csv(os.path.join(out_dir, out_top1_basename), sep='\t', index=False)

    return


if __name__ == "__main__":
    annotate_pseudo_ms2_spec(
        '/demo/lc_ms/single_files/NIST_pool_1_20eV_pseudo_ms2.mgf',
        ['/Users/shipei/Documents/projects/ms1_id/data/gnps.pkl',
         '/Users/shipei/Documents/projects/ms1_id/data/gnps_k10.pkl'])
