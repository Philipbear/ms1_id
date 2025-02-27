import os
import numpy as np
import pandas as pd


def write_ms1_id_results(feature_ls, ms2_spec_ls, save=True, save_dir=None):
    """
    Output the annotated spectra
    :param feature_ls: a list of MSiFeature object
    :param ms2_spec_ls: a list of PseudoMS2-like object
    :param save: bool, whether to save the results
    :param save_dir: str, path to save the results
    :return: None
    """

    # only write out spectra with annotations
    ms2_spec_ls = [spec for spec in ms2_spec_ls if spec.annotated]

    mz_values = np.array([feature.mz for feature in feature_ls])
    spatial_chaos = np.array([feature.spatial_chaos for feature in feature_ls])

    out_list = []
    for spec in ms2_spec_ls:
        # pseudo_ms2_str: mz1 int1; mz2 int2; ...
        pseudo_ms2_str = ' '.join([f"{mz:.4f} {intensity:.0f};" for mz, intensity in zip(spec.mzs, spec.intensities)])

        for annotation in spec.annotation_ls:

            # find the idx of the closest mz and get the corresponding spatial chaos
            t_mz_idx = np.argmin(np.abs(mz_values - annotation.mz))
            spatial_chaos_value = spatial_chaos[t_mz_idx]

            out_list.append({
                'pms2_idx': spec.spec_idx,
                'name': annotation.name,
                'mz': round(annotation.mz, 4),
                'spatial_chaos':  round(spatial_chaos_value, 4),
                'matched_score': round(annotation.score, 4),
                'matched_peak': annotation.matched_peak,
                'spectral_usage': round(annotation.spectral_usage, 4) if annotation.spectral_usage else None,
                # 'search_eng_matched_id': annotation.search_eng_matched_id,
                'precursor_mz': round(annotation.precursor_mz, 4),
                'precursor_type': annotation.precursor_type,
                'formula': annotation.formula,
                'inchikey': annotation.inchikey,
                # 'instrument_type': annotation.instrument_type,
                'collision_energy': annotation.collision_energy,
                'db_name': annotation.db_name,
                'db_id': annotation.db_id,
                'pseudo_ms2': pseudo_ms2_str,
            })

    out_df = pd.DataFrame(out_list)

    if save and save_dir:
        save_path = os.path.join(save_dir, 'ms1_id_annotations_all.tsv')
        out_df.to_csv(save_path, index=False, sep='\t')

        # save a dereplicated version
        if 'matched_score' in out_df.columns:
            # sort
            out_df = out_df.sort_values(['spectral_usage', 'matched_score', 'matched_peak'],
                                        ascending=[False, False, False])

            # dereplicate by [inchikey, rounded precursor mz]  # rounded precursor mz indicating precursor type
            out_df['rounded_precursor_mz'] = out_df['precursor_mz'].round(2)
            out_df['2d_inchikey'] = out_df['inchikey'].str[:14]
            out_df = out_df.drop_duplicates(['2d_inchikey', 'rounded_precursor_mz'], keep='first')
            out_df.drop(columns=['rounded_precursor_mz'], inplace=True)

            out_df.to_csv(save_path.replace('_all.tsv', '_derep.tsv'), index=False, sep='\t')

    return out_df


def write_pseudoms2_to_mgf(pseudo_ms2_spectra, mgf_path):
    """
    Write pseudo MS2 spectra to an MGF file
    :param pseudo_ms2_spectra: list of PseudoMS2 objects
    :param mgf_path: str, path to save the MGF file
    :return: None
    """
    with open(mgf_path, 'w') as f:
        idx = 1
        for spec in pseudo_ms2_spectra:
            f.write(f"BEGIN IONS\n")
            f.write(f"PEPMASS={round(spec.t_mz, 5)}\n")
            f.write(f"SCANS={idx}\n")
            for mz, intensity in zip(spec.mzs, spec.intensities):
                f.write(f"{mz:.5f} {intensity:.0f}\n")
            f.write(f"END IONS\n\n")
            idx += 1
    return None