import os
import pandas as pd
import numpy as np
from _utils import AlignedMS1Annotation


def write_ms1_id_results(ms1_spec_ls, save=True, out_dir=None):
    """
    Output the annotated ms1 spectra
    :param ms1_spec_ls: a list of PseudoMS1-like object
    :param save: bool, whether to save the results
    :param out_dir: output folder
    :return: None
    """

    # only write out spectra with annotations
    ms1_spec_ls = [spec for spec in ms1_spec_ls if spec.annotated]

    out_list = []
    for spec in ms1_spec_ls:
        # pseudo_ms1_str: mz1 int1; mz2 int2; ...
        pseudo_ms1_str = ' '.join([f"{mz:.4f} {intensity:.0f};" for mz, intensity in zip(spec.mzs, spec.intensities)])

        for annotation in spec.annotation_ls:

            annotation_peaks = annotation.peaks
            annotation_peaks = annotation_peaks[annotation_peaks[:, 1] > 0]  # remove zero intensity peaks
            matched_peak_str = ' '.join([f"{mz:.4f} {intensity:.0f};" for mz, intensity in annotation_peaks])

            out_list.append({
                'file_name': spec.file_name,
                'rt': round(spec.rt, 2) if spec.rt else None,
                'intensity': round(annotation.intensity, 0) if annotation.intensity else None,
                'name': annotation.name,
                'precursor_mz': round(annotation.precursor_mz, 4),
                'matched_score': round(annotation.score, 4),
                'matched_peak': annotation.matched_peak,
                'spectral_usage': round(annotation.spectral_usage, 4) if annotation.spectral_usage else None,
                # 'search_eng_matched_id': annotation.search_eng_matched_id,
                'precursor_type': annotation.precursor_type,
                'formula': annotation.formula,
                'inchikey': annotation.inchikey,
                'instrument_type': annotation.instrument_type,
                'collision_energy': annotation.collision_energy,
                'pseudo_ms1': pseudo_ms1_str,
                'matched_spectrum': matched_peak_str
            })

    out_df = pd.DataFrame(out_list)

    if save:
        out_path = os.path.join(out_dir, 'ms1_id_results.tsv')
        out_df.to_csv(out_path, index=False, sep='\t')

    return out_df


def write_single_file(msdata, pseudo_ms1_spectra=None, ion_mode='positive', save_path=None):
    """
    Function to generate a report for rois in csv format.
    """
    common_adduct_list = ['[M+H]+', 'M+H', '[M+H-H2O]+', 'M+H-H2O', '[M-H2O+H]+', 'M-H2O+H', '[M+NH4]+', 'M+NH4',
                          '[M]+', 'M+', 'Cat', 'Cat+'] if ion_mode == 'positive' else ['[M-H]-', 'M-H']

    result = []

    for roi in msdata.rois:
        iso_dist = ""
        for i in range(len(roi.isotope_mz_seq)):
            iso_dist += (str(np.round(roi.isotope_mz_seq[i], decimals=4)) + ";" +
                         str(np.round(roi.isotope_int_seq[i], decimals=0)) + "|")
        iso_dist = iso_dist[:-1]

        ms2 = ""
        if roi.best_ms2 is not None:
            for i in range(len(roi.best_ms2.peaks)):
                ms2 += (str(np.round(roi.best_ms2.peaks[i, 0], decimals=4)) + ";" +
                        str(np.round(roi.best_ms2.peaks[i, 1], decimals=0)) + "|")
            ms2 = ms2[:-1]

        temp = [roi.id, roi.mz.__round__(4), roi.rt.__round__(3), roi.length, roi.rt_seq[0],
                roi.rt_seq[-1], roi.peak_area, roi.peak_height, roi.gaussian_similarity.__round__(2),
                roi.noise_level.__round__(2), roi.asymmetry_factor.__round__(2), roi.charge_state, roi.is_isotope,
                str(roi.isotope_id_seq)[1:-1], iso_dist,
                # roi.is_in_source_fragment, roi.isf_parent_roi_id, str(roi.isf_child_roi_id)[1:-1],
                roi.adduct_type,
                # roi.adduct_parent_roi_id, str(roi.adduct_child_roi_id)[1:-1],
                ]

        temp.extend([ms2, roi.annotation, roi.formula, roi.similarity, roi.matched_peak_number,
                     roi.inchikey])

        result.append(temp)

    # convert result to a pandas dataframe
    columns = ["ID", "m/z", "RT", "length", "RT_start", "RT_end", "peak_area", "peak_height",
               "Gaussian_similarity", "noise_level", "asymmetry_factor", "charge", "is_isotope",
               "isotope_IDs", "isotopes",
               # "is_in_source_fragment", "ISF_parent_ID", "ISF_child_ID",
               "adduct",
               # "adduct_base_ID", "adduct_other_ID"
               ]

    columns.extend(["MS2", "MS2_annotation", "MS2_formula", "MS2_similarity", "MS2_matched_peak", "MS2_inchikey"])

    df = pd.DataFrame(result, columns=columns)

    # add ms1 id results
    ms1_columns = ['MS1_annotation', 'MS1_formula', 'MS1_similarity', 'MS1_matched_peak',
                   'MS1_spectral_usage', 'MS1_precursor_type', 'MS1_inchikey', 'MS1_collision_energy']
    df = df.reindex(columns=df.columns.tolist() + ms1_columns)

    if pseudo_ms1_spectra is not None:
        ms1_spec_ls = [spec for spec in pseudo_ms1_spectra if spec.annotated]

        if len(ms1_spec_ls) > 0:
            for spec in ms1_spec_ls:
                for annotation in spec.annotation_ls:
                    this_precmz = annotation.precursor_mz
                    # Create a boolean mask for the conditions
                    mask = ((df['m/z'] - this_precmz).abs() <= msdata.params.mz_tol_ms1) & (
                                (df['RT'] - spec.rt).abs() <= 1)

                    if not mask.any():
                        continue

                    idx = df.loc[mask, 'RT'].sub(spec.rt).abs().idxmin()

                    # if the annotation is better than the previous one, update the row
                    new_values = {
                        'MS1_annotation': annotation.name,
                        'MS1_formula': annotation.formula,
                        'MS1_similarity': round(float(annotation.score), 3),
                        'MS1_matched_peak': annotation.matched_peak,
                        'MS1_spectral_usage': round(float(annotation.spectral_usage), 4),
                        'MS1_precursor_type': annotation.precursor_type,
                        'MS1_inchikey': annotation.inchikey,
                        'MS1_collision_energy': annotation.collision_energy
                    }

                    current_values = df.loc[idx, ms1_columns]

                    # if the current row is empty
                    # if common adduct, compare similarity and matched peak number
                    if pd.isna(current_values['MS1_similarity']) or \
                            (annotation.precursor_type in common_adduct_list and
                             (current_values['MS1_precursor_type'] not in common_adduct_list or
                              (annotation.score >= current_values['MS1_similarity'] and
                               annotation.matched_peak >= current_values['MS1_matched_peak']))):
                        df.loc[idx, ms1_columns] = new_values

    # save the dataframe to csv file
    df.to_csv(save_path, index=False, sep="\t")


def write_feature_table(df, pseudo_ms1_spectra, config, output_path):
    """
    A function to output the aligned feature table.
    :param df: pd.DataFrame, feature table
    :param pseudo_ms1_spectra: list of PseudoMS1-like object
    :param config: Parameters
    :param output_path: str, output file path
        columns=["ID", "m/z", "RT", "adduct", "is_isotope", "is_in_source_fragment", "Gaussian_similarity", "noise_level",
             "asymmetry_factor", "charge", "isotopes", "MS2", "matched_MS2", "search_mode",
             "annotation", "formula", "similarity", "matched_peak_number", "SMILES", "InChIKey",
                "fill_percentage", "alignment_reference"] + sample_names
    """

    # keep four digits for the m/z column and three digits for the RT column
    df["m/z"] = df["m/z"].apply(lambda x: round(x, 4))
    df["RT"] = df["RT"].apply(lambda x: round(x, 3))
    df['fill_percentage'] = df['fill_percentage'].apply(lambda x: round(x, 2))
    df['similarity'] = df['similarity'].astype(float)
    df['similarity'] = df['similarity'].apply(lambda x: round(x, 4))

    # refine ms1 id results with feature table. for each feature, choose the most confident annotation
    ms1_annotation_ls = _refine_pseudo_ms1_spectra_list(pseudo_ms1_spectra, df, config)

    # add ms1 id results
    df['MS1_annotation'] = None
    df['MS1_formula'] = None
    df['MS1_similarity'] = None
    df['MS1_matched_peak'] = None
    df['MS1_spectral_usage'] = None
    df['MS1_precursor_type'] = None
    df['MS1_inchikey'] = None
    df['MS1_collision_energy'] = None

    # add ms1 id results to the feature table
    for aligned_ms1_annotation in ms1_annotation_ls:
        idx = aligned_ms1_annotation.df_idx
        annotation = aligned_ms1_annotation.selected_annotation
        df.loc[idx, 'MS1_annotation'] = annotation.name
        df.loc[idx, 'MS1_formula'] = annotation.formula
        df.loc[idx, 'MS1_similarity'] = round(float(annotation.score), 3)
        df.loc[idx, 'MS1_matched_peak'] = annotation.matched_peak
        df.loc[idx, 'MS1_spectral_usage'] = round(float(annotation.spectral_usage), 4)
        df.loc[idx, 'MS1_precursor_type'] = annotation.precursor_type
        df.loc[idx, 'MS1_inchikey'] = annotation.inchikey
        df.loc[idx, 'MS1_collision_energy'] = annotation.collision_energy

    df.to_csv(output_path, index=False, sep="\t")


def _refine_pseudo_ms1_spectra_list(pseudo_ms1_spectra, df, config):
    """
    For each feature, choose the most confident annotation, prioritizing common adducts.

    :param pseudo_ms1_spectra: list of PseudoMS1-like objects
    :param df: feature table
    :param config: configuration object containing align_mz_tol and common_adduct_list
    :return: list of AlignedMS1Annotation objects with selected annotations
    """

    aligned_ms1_annotations = {}  # Dictionary to store AlignedMS1Annotation objects, keyed by df index

    for spec in (s for s in pseudo_ms1_spectra if s.annotated):
        for annotation in spec.annotation_ls:
            mask = ((df['m/z'] - annotation.precursor_mz).abs() <= config.align_mz_tol) & \
                   ((df['RT'] - spec.rt).abs() <= 1)

            if not mask.any():
                continue

            idx = df.loc[mask, 'RT'].sub(spec.rt).abs().idxmin()

            if idx not in aligned_ms1_annotations:
                aligned_ms1_annotations[idx] = AlignedMS1Annotation(idx)

            aligned_ms1_annotations[idx].annotation_ls.append(annotation)

    # Select the best annotation for each aligned MS1 feature
    for aligned_ms1_annotation in aligned_ms1_annotations.values():
        common_adduct_annotations = [a for a in aligned_ms1_annotation.annotation_ls
                                     if a.precursor_type in config.common_adduct_list]

        if common_adduct_annotations:
            aligned_ms1_annotation.selected_annotation = max(common_adduct_annotations, key=lambda x: x.score)
        else:
            aligned_ms1_annotation.selected_annotation = max(aligned_ms1_annotation.annotation_ls,
                                                             key=lambda x: x.score)

    return list(aligned_ms1_annotations.values())