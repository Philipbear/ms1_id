# Description: This file contains classes that are used in the main pipeline.
import numpy as np


class Spectrum:
    def __init__(self, mz_ls, int_ls):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.annotation_ls = []  # list of SpecAnnotation objects


class PseudoMS2:

    def __init__(self, t_mz, mz_ls, int_ls, roi_ids, file_name, rt, mz_tol):
        self.t_mz = t_mz  # this PseudoMS2 spectrum is generated starting from this mz
        self.mzs = mz_ls
        self.intensities = int_ls
        self.roi_ids = roi_ids
        self.file_name = file_name
        self.rt = rt
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects

        # find the index of the target m/z in the pseudo MS2 spectrum, with the closest m/z
        t_mz_idx = np.argmin(np.abs(np.array(self.mzs) - self.t_mz))
        if abs(mz_ls[t_mz_idx] - self.t_mz) <= mz_tol:
            self.t_mz_idx = t_mz_idx
            self.t_mz_intensity = self.intensities[t_mz_idx]
        else:
            self.t_mz_idx = None  # if the target m/z is not in the pseudo MS2 spectrum
            self.t_mz_intensity = 0.0


class SpecAnnotation:
    def __init__(self, pseudo_ms2_spec_mzs, pseudo_ms2_spec_intensities,
                 prec_mz, db_name, idx, score, matched_peak, spectral_usage, mz_tol):
        self.db_name = db_name
        self.search_eng_matched_id = idx  # index of the matched spec in the search engine
        self.score = score
        self.matched_peak = matched_peak
        self.spectral_usage = spectral_usage
        self.matched_spec = None  # the matched spectrum in the search engine
        self.db_id = None
        self.name = None
        self.precursor_mz = prec_mz
        self.precursor_type = None
        self.formula = None
        self.inchikey = None
        self.instrument_type = None
        self.collision_energy = None
        self.centroided_peaks = None

        # annotation in this pseudo MS2 spectrum
        t_mz_idx = np.argmin(np.abs(np.array(pseudo_ms2_spec_mzs) - self.precursor_mz))
        if abs(pseudo_ms2_spec_mzs[t_mz_idx] - self.precursor_mz) <= mz_tol:
            self.pseudo_ms2_precursor_mz_idx = t_mz_idx
            self.pseudo_ms2_precursor_intensity = pseudo_ms2_spec_intensities[t_mz_idx]
        else:
            self.pseudo_ms2_precursor_mz_idx = None
            self.pseudo_ms2_precursor_intensity = 0.0

    def __str__(self):
        return f"name {self.name}, score {self.score}, matched_peak {self.matched_peak}, db_id {self.db_id}"


class AnnotatedPseudoMS2:
    """
    An annotation with pseudo MS2 spectrum stored
    """
    def __init__(self, spec_annotation, mzs, intensities):
        self.annotation = spec_annotation
        self.pseudo_ms2_mzs = mzs
        self.pseudo_ms2_intensities = intensities


class AlignedMS1Annotation:
    """
    for exporting aligned feature table with MS1 IDs
    store all possible MS1 IDs for a feature
    """

    def __init__(self, idx):
        self.df_idx = idx  # index of the matched feature in the feature table
        self.annotated_pseudo_ms2_list = []  # list of AnnotatedPseudoMS2 objects
        self.selected_annotated_pseudo_ms2 = None


def find_ms_info(file_name):
    """
    Find the type of MS and ion mode from the raw file.
    """

    ms_type = 'tof'
    ion_mode = 'positive'
    centroid = False

    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            if 'orbitrap' in line.lower() or 'Q Exactive' in line.lower():
                ms_type = 'orbitrap'
            if 'negative' in line.lower() or 'polarity="-"' in line.lower():
                ion_mode = 'negative'
            if "centroid spectrum" in line.lower() or 'centroided="1"' in line.lower():
                centroid = True
            if i > 200:
                break

    return ms_type, ion_mode, centroid