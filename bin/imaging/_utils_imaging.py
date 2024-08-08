# Description: This file contains classes that are used in the main pipeline.


class PseudoMS1:
    def __init__(self, mz_ls, int_ls, idx_ls):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.indices = idx_ls  # indices of mzs, for later assign intensities
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects


class SpecAnnotation:
    def __init__(self, idx, score, matched_peak):
        self.search_eng_matched_id = idx  # index of the matched spec in the search engine
        self.score = score
        self.matched_peak = matched_peak
        self.spectral_usage = None
        self.matched_spec = None  # the matched spectrum in the search engine
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

