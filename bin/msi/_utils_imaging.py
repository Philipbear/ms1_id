# Description: This file contains classes that are used in the main pipeline.


class PseudoMS2:
    def __init__(self, t_mz, t_mz_idx, mz_ls, int_ls, idx_ls):
        self.t_mz = t_mz  # this PseudoMS1 spectrum is generated starting from this mz
        self.t_mz_idx = t_mz_idx  # idx of t_mz in the mz_values
        self.mzs = mz_ls
        self.intensities = int_ls
        self.indices = idx_ls  # indices of mzs, for later assign intensities
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects


class SpecAnnotation:
    def __init__(self, db_name, idx, score, matched_peak):
        self.db_name = db_name
        self.search_eng_matched_id = idx  # index of the matched spec in the search engine
        self.score = score
        self.matched_peak = matched_peak
        self.spectral_usage = None
        self.matched_spec = None  # the matched spectrum in the search engine
        self.db_id = None
        self.name = None
        self.mz = None  # mz in PseudoMS2
        self.precursor_mz = None  # precursor mz of matched spec
        self.precursor_type = None
        self.formula = None
        self.inchikey = None
        self.instrument_type = None
        self.collision_energy = None
        self.centroided_peaks = None

    def __str__(self):
        return f"name {self.name}, score {self.score}, matched_peak {self.matched_peak}, db_id {self.db_id}"
