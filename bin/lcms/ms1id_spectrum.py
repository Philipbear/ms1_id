from _reverse_matching import prepare_ms2_lib, ms1_id_annotation
from _utils import Spectrum


def prepare_ms2_db(msp_path, mz_tol=0.01, peak_scale_k=10, peak_intensity_power=0.5):
    """
    Prepare the MS2 library (pickle file) from the msp file
    :param msp_path: path to the msp file
    :param mz_tol: float, mz tolerance
    :param peak_scale_k: float, peak scale factor for intensity scaling (only for reference spectra)
    :param peak_intensity_power: float, peak intensity power
    :return: search engine object
    """
    prepare_ms2_lib(ms2db=msp_path, mz_tol=mz_tol,
                    peak_scale_k=peak_scale_k, peak_intensity_power=peak_intensity_power)

    return


def annotate_ms1_id(ms1_spectra,
                    msms_library_path,
                    mz_tol=0.01,
                    score_cutoff=0.8,
                    min_matched_peak=6):
    """
    Perform ms1 annotation
    :param ms1_spectra: a list of PseudoMS1-like object
    :param msms_library_path: path to the msms library
    :param mz_tol: float, mz tolerance
    :param rt_tol: float, rt tolerance
    :param score_cutoff: float, score cutoff
    :param min_matched_peak: int, minimum matched peak
    :param min_prec_int_in_ms1: minimum precursor intensity in MS1 spectrum
    :param max_prec_rel_int_in_other_ms2: float, maximum relative intensity in the precursor ion
    :return: annotated ms1_spectra
    """

    return ms1_id_annotation(ms1_spec_ls=ms1_spectra,
                             ms2_library=msms_library_path,
                             mz_tol=mz_tol,
                             score_cutoff=score_cutoff,
                             min_matched_peak=min_matched_peak,
                             refine=False)


if __name__ == "__main__":
    # example usage
    prepare_ms2_db(msp_path='data/MSMS.msp', mz_tol=0.01)

    # prepare query spectra
    ms1_spectra = [
        Spectrum(mz_ls=[100, 200, 300], int_ls=[100, 200, 300]),
        Spectrum(mz_ls=[200, 500, 600], int_ls=[200, 500, 600])
    ]

    # annotate spectra
    annotated_ms1_spectra = annotate_ms1_id(ms1_spectra=ms1_spectra,
                                            msms_library_path='data/MSMS.pkl',
                                            mz_tol=0.01,
                                            score_cutoff=0.8,
                                            min_matched_peak=6)

    # print the results
    for spec in annotated_ms1_spectra:
        print(spec.annotation_ls)
