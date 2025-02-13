import os
import pandas as pd
import numpy as np
from ms1_id.lcms.utils import AlignedMS1Annotation, AnnotatedPseudoMS2
from ms1_id.lcms.centroid_data import consensus_spectrum


# def write_ms1_id_results(ms1_spec_ls, out_dir=None):
#     """
#     Output the annotated ms1 spectra
#     :param ms1_spec_ls: a list of PseudoMS2-like object
#     :param out_dir: output folder
#     :return: None
#     """
#
#     # only write out spectra with annotations
#     ms1_spec_ls = [spec for spec in ms1_spec_ls if spec.annotated]
#
#     out_list = []
#     for spec in ms1_spec_ls:
#         # pseudo_ms2_str: mz1 int1; mz2 int2; ...
#         pseudo_ms2_str = ''.join([f"{mz:.4f} {intensity:.0f};" for mz, intensity in zip(spec.mzs, spec.intensities)])
#
#         for annotation in spec.annotation_ls:
#             out_list.append({
#                 'file_name': spec.file_name,
#                 'rt': round(spec.rt, 2) if spec.rt else None,
#                 'db_name': annotation.db_name,
#                 'name': annotation.name,
#                 'precursor_mz': round(annotation.precursor_mz, 4),
#                 'matched_score': round(annotation.score, 4),
#                 'matched_peak': annotation.matched_peak,
#                 'spectral_usage': round(annotation.spectral_usage, 4) if annotation.spectral_usage else None,
#                 # 'search_eng_matched_id': annotation.search_eng_matched_id,
#                 'precursor_type': annotation.precursor_type,
#                 'formula': annotation.formula,
#                 'inchikey': annotation.inchikey,
#                 'instrument_type': annotation.instrument_type,
#                 'collision_energy': annotation.collision_energy,
#                 'db_id': annotation.db_id,
#                 'pseudo_ms2': pseudo_ms2_str,
#             })
#
#     out_df = pd.DataFrame(out_list)
#
#     out_path = os.path.join(out_dir, 'ms1_id_results.tsv')
#     out_df.to_csv(out_path, index=False, sep='\t')
#
#     return out_df


def write_single_file(msdata, pseudo_ms2_spectra=None, save_path=None):
    """
    Function to generate a report for rois in csv format.
    """

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
                     roi.inchikey, getattr(roi, 'precursor_type', None), getattr(roi, 'collision_energy', None)])

        result.append(temp)

    # convert result to a pandas dataframe
    columns = ["ID", "m/z", "RT", "length", "RT_start", "RT_end", "peak_area", "peak_height",
               "Gaussian_similarity", "noise_level", "asymmetry_factor", "charge", "is_isotope",
               "isotope_IDs", "isotopes",
               # "is_in_source_fragment", "ISF_parent_ID", "ISF_child_ID",
               "adduct",
               # "adduct_base_ID", "adduct_other_ID"
               ]

    columns.extend(["MS2", "MS2_annotation", "MS2_formula", "MS2_similarity", "MS2_matched_peak",
                    "MS2_inchikey", "MS2_precursor_type", "MS2_collision_energy"])

    df = pd.DataFrame(result, columns=columns)

    # add ms1 id results
    df['MS1_annotation'] = None
    df['MS1_formula'] = None
    df['MS1_similarity'] = None
    df['MS1_matched_peak'] = None
    df['MS1_spectral_usage'] = None
    df['MS1_precursor_type'] = None
    df['MS1_inchikey'] = None
    df['MS1_collision_energy'] = None
    df['MS1_db_name'] = None
    df['MS1_db_id'] = None
    df['pseudo_ms2'] = None

    if pseudo_ms2_spectra is not None:
        spec_ls = [spec for spec in pseudo_ms2_spectra if spec.annotated]

        if len(spec_ls) > 0:
            for spec in spec_ls:

                pseudo_ms2_str = ''.join(
                    [f"{mz:.4f} {intensity:.0f};" for mz, intensity in zip(spec.mzs, spec.intensities)])

                for annotation in spec.annotation_ls:
                    this_precmz = annotation.precursor_mz
                    # Create a boolean mask for the conditions
                    mask = (((df['m/z'] - this_precmz).abs() <= msdata.params.mz_tol_ms1) &
                            ((df['RT'] - spec.rt).abs() <= 0.1))

                    if not mask.any():
                        continue

                    idx = df.loc[mask, 'RT'].sub(spec.rt).abs().idxmin()

                    # if the annotation is better than the previous one, update the row
                    if df.loc[idx, 'MS1_similarity'] is None or annotation.score >= df.loc[idx, 'MS1_similarity']:
                        df.loc[idx, 'MS1_annotation'] = annotation.name
                        df.loc[idx, 'MS1_formula'] = annotation.formula
                        df.loc[idx, 'MS1_similarity'] = round(float(annotation.score), 4)
                        df.loc[idx, 'MS1_matched_peak'] = annotation.matched_peak
                        df.loc[idx, 'MS1_spectral_usage'] = round(float(annotation.spectral_usage), 4)
                        df.loc[idx, 'MS1_precursor_type'] = annotation.precursor_type
                        df.loc[idx, 'MS1_inchikey'] = annotation.inchikey
                        df.loc[idx, 'MS1_collision_energy'] = annotation.collision_energy
                        df.loc[idx, 'MS1_db_name'] = annotation.db_name
                        df.loc[idx, 'MS1_db_id'] = annotation.db_id
                        df.loc[idx, 'pseudo_ms2'] = pseudo_ms2_str

        # # for unannotated features, fill in the pseudo MS2 spectra
        # df = fill_pseudo_ms2_spectra_for_unannotated_features(df, pseudo_ms2_spectra, msdata.params)

    # save the dataframe to csv file
    df.to_csv(save_path, index=False, sep="\t")


def write_feature_table(df, pseudo_ms2_spectra, config, output_path):
    """
    A function to output the aligned feature table.
    :param df: pd.DataFrame, feature table
    :param pseudo_ms2_spectra: list of PseudoMS2-like object
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
    aligned_ms1_annotation_ls = refine_pseudo_ms2_spectra_list(pseudo_ms2_spectra, df, config)

    # Add MS1 id results columns
    ms1_columns = ['MS1_annotation', 'MS1_formula', 'MS1_similarity', 'MS1_matched_peak',
                   'MS1_spectral_usage', 'MS1_precursor_type', 'MS1_inchikey',
                   'MS1_collision_energy', 'MS1_db_name', 'MS1_db_id', 'pseudo_ms2']
    for col in ms1_columns:
        df[col] = None

    # add ms1 id results to the feature table
    df = add_ms1_id_results(df, aligned_ms1_annotation_ls)

    # for unannotated features, fill in the pseudo MS2 spectra
    df = fill_pseudo_ms2_spectra_for_unannotated_features(df, pseudo_ms2_spectra, config)

    df.to_csv(output_path, index=False, sep="\t")


def add_ms1_id_results(df, aligned_ms1_annotation_ls):
    """
    Add MS1 ID results to the feature table
    """
    for aligned_ms1_annotation in aligned_ms1_annotation_ls:
        idx = aligned_ms1_annotation.df_idx

        annotation = aligned_ms1_annotation.selected_annotated_pseudo_ms2.annotation
        df.loc[idx, 'MS1_annotation'] = annotation.name
        df.loc[idx, 'MS1_formula'] = annotation.formula
        df.loc[idx, 'MS1_similarity'] = round(float(annotation.score), 4)
        df.loc[idx, 'MS1_matched_peak'] = annotation.matched_peak
        df.loc[idx, 'MS1_spectral_usage'] = round(float(annotation.spectral_usage), 4)
        df.loc[idx, 'MS1_precursor_type'] = annotation.precursor_type
        df.loc[idx, 'MS1_inchikey'] = annotation.inchikey
        df.loc[idx, 'MS1_collision_energy'] = annotation.collision_energy
        df.loc[idx, 'MS1_db_name'] = annotation.db_name
        df.loc[idx, 'MS1_db_id'] = annotation.db_id

        pseudo_ms2_str = ''.join(
            [f"{mz:.4f} {intensity:.0f};" for mz, intensity in
             zip(aligned_ms1_annotation.selected_annotated_pseudo_ms2.pseudo_ms2_mzs,
                 aligned_ms1_annotation.selected_annotated_pseudo_ms2.pseudo_ms2_intensities)])
        df.loc[idx, 'pseudo_ms2'] = pseudo_ms2_str

    return df


def fill_pseudo_ms2_spectra_for_unannotated_features(df, pseudo_ms2_spectra, config):
    """
    Fill in the pseudo MS2 spectra for unannotated features
    """
    for spec in pseudo_ms2_spectra:
        # Create a boolean mask for the conditions
        mask = (((df['m/z'] - spec.t_mz).abs() <= config.align_mz_tol) &
                ((df['RT'] - spec.rt).abs() <= config.align_rt_tol))

        if not mask.any():
            continue

        # Find the row with the smallest RT difference
        idx = df.loc[mask, 'RT'].sub(spec.rt).abs().idxmin()

        # if already annotated, skip
        if df.loc[idx, 'MS1_similarity'] is not None:
            continue

        pseudo_ms2_peaks = np.column_stack((np.array(spec.mzs), np.array(spec.intensities)))

        if df.loc[idx, 'pseudo_ms2'] is None:
            df.loc[idx, 'pseudo_ms2'] = [pseudo_ms2_peaks]
        else:
            df.loc[idx, 'pseudo_ms2'].append(pseudo_ms2_peaks)

    # consensus pseudo MS2 spectra for unannotated features
    for idx, row in df.iterrows():
        if row['MS1_similarity'] is None and row['pseudo_ms2'] is not None:
            pseudo_ms2_peaks_ls = row['pseudo_ms2']

            merged_peaks = np.concatenate(pseudo_ms2_peaks_ls, axis=0)

            # find the consensus pseudo MS2 spectrum
            consensus_peaks = consensus_spectrum(merged_peaks, config.mz_tol_ms2)

            pseudo_ms2_str = ''.join([f"{mz:.4f} {intensity:.0f};" for mz, intensity in consensus_peaks])

            # add the consensus pseudo MS2 spectrum to the feature table
            df.loc[idx, 'pseudo_ms2'] = pseudo_ms2_str

    return df


def refine_pseudo_ms2_spectra_list(pseudo_ms2_spectra, df, config):
    """
    for each feature, choose the most confident annotation
    :return: AlignedMS1Annotation with selected annotations
    """

    # only reserve annotated pseudo MS2 spectra
    spec_ls = [spec for spec in pseudo_ms2_spectra if spec.annotated]

    all_df_idx_ls = []  # list of indices in the feature table that have been matched
    aligned_ms1_annotation_ls = []  # list of AlignedMS1Annotation objects

    for spec in spec_ls:
        this_rt = spec.rt
        for annotation in spec.annotation_ls:
            this_precmz = annotation.precursor_mz
            # Create a boolean mask for the conditions
            mask = ((df['m/z'] - this_precmz).abs() <= config.align_mz_tol) & (
                        (df['RT'] - this_rt).abs() <= config.align_rt_tol)

            if not mask.any():
                continue

            # Find the row with the smallest RT difference
            idx = df.loc[mask, 'RT'].sub(this_rt).abs().idxmin()

            annotated_pseudo_ms2 = AnnotatedPseudoMS2(annotation, spec.mzs, spec.intensities)

            if idx in all_df_idx_ls:
                aligned_ms1_annotation_idx = all_df_idx_ls.index(idx)
                # add annotation to the existing AlignedMS1Annotation object
                aligned_ms1_annotation_ls[aligned_ms1_annotation_idx].annotated_pseudo_ms2_list.append(
                    annotated_pseudo_ms2)
            else:
                # add the index to the list
                all_df_idx_ls.append(idx)
                # create an AlignedMS1Annotation object
                aligned_ms1_annotation = AlignedMS1Annotation(idx)
                aligned_ms1_annotation.annotated_pseudo_ms2_list.append(annotated_pseudo_ms2)

                # add the object to the total list
                aligned_ms1_annotation_ls.append(aligned_ms1_annotation)

    # find the one with the highest similarity score
    for aligned_ms1_annotation in aligned_ms1_annotation_ls:
        if len(aligned_ms1_annotation.annotated_pseudo_ms2_list) > 1:
            aligned_ms1_annotation.selected_annotated_pseudo_ms2 = (
                max(aligned_ms1_annotation.annotated_pseudo_ms2_list, key=lambda x: x.annotation.score))
        else:
            aligned_ms1_annotation.selected_annotated_pseudo_ms2 = aligned_ms1_annotation.annotated_pseudo_ms2_list[0]

    return aligned_ms1_annotation_ls


def write_pseudoms2_to_mgf(pseudoms2_ls, save_dir, file_name):
    """
    Write pseudo MS2 spectra to MGF file
    """

    mgf_path = os.path.join(save_dir, f"{file_name}_pseudo_ms2.mgf")

    with open(mgf_path, 'w') as f:
        idx = 1
        for spec in pseudoms2_ls:
            f.write(f"BEGIN IONS\n")
            # f.write(f"PEPMASS={round(spec.t_mz, 5)}\n")
            f.write("PEPMASS=0")
            f.write(f"SCANS={idx}\n")
            f.write(f"RTINSECONDS={spec.rt * 60}\n")

            mz_arr = np.array(spec.mzs)
            intensity_arr = np.array(spec.intensities)

            # sort by mz
            sort_idx = np.argsort(mz_arr)
            mz_arr = mz_arr[sort_idx]
            intensity_arr = intensity_arr[sort_idx]

            for i in range(len(mz_arr)):
                f.write(f"{mz_arr[i]:.5f} {intensity_arr[i]:.0f}\n")

            f.write(f"END IONS\n\n")
            idx += 1
    return
