class Spectrum:
    def __init__(self, mz_ls, int_ls):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.annotation_ls = []  # list of SpecAnnotation objects


class PseudoMS1:
    def __init__(self, mz_ls, int_ls, feature_ids, rt):
        self.mzs = mz_ls
        self.intensities = int_ls
        self.feature_ids = feature_ids
        self.rt = rt
        self.annotated = False
        self.annotation_ls = []  # list of SpecAnnotation objects


class SpecAnnotation:
    def __init__(self, score, matched_peak):
        self.score = score
        self.matched_peak = matched_peak
        self.db_id = None
        self.name = None
        self.precursor_type = None
        self.formula = None
        self.inchikey = None
        self.instrument_type = None
        self.collision_energy = None

    def __str__(self):
        return f"name {self.name}, score {self.score}, matched_peak {self.matched_peak}, db_id {self.db_id}"
