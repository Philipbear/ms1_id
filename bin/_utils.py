# Description: This file contains classes that are used in the main pipeline.

class Spectrum:
    def __init__(self, mz_ls, int_ls):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.annotation_ls = []  # list of SpecAnnotation objects


class PseudoMS1:
    def __init__(self, mz_ls, int_ls, roi_ids, file_name, rt):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.roi_ids = roi_ids
        self.file_name = file_name
        self.rt = rt
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects


class SpecAnnotation:
    def __init__(self, idx, score, matched_peak):
        self.search_eng_matched_id = idx  # index of the matched spec in the search engine
        self.score = score
        self.matched_peak = matched_peak
        self.spectral_usage = None
        self.intensity = None  # intensity of the precursor ion in PseudoMS1 spectrum
        self.db_id = None
        self.name = None
        self.precursor_mz = None
        self.precursor_type = None
        self.formula = None
        self.inchikey = None
        self.instrument_type = None
        self.collision_energy = None
        self.peaks = None

    def __str__(self):
        return f"name {self.name}, score {self.score}, matched_peak {self.matched_peak}, db_id {self.db_id}"


class RoiPair:
    def __init__(self, roi_a, roi_b, ppc):

        # compare id
        if roi_a.id < roi_b.id:
            self.roi_1 = roi_a
            self.roi_2 = roi_b
        else:
            self.roi_1 = roi_b
            self.roi_2 = roi_a

        self.ppc = ppc  # peak-peak correlation
