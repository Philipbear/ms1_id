from bin.lcms._reverse_matching import prepare_ms2_lib


def prepare_ms2db(msp_path,
                  mz_tol: float = 0.05,
                  peak_scale_k: float = None,
                  peak_intensity_power: float = 0.5):
    """
    Index MS2 library from the msp file and save to a pickle file
    :param msp_path: path to the msp file
    :param mz_tol: float, mz tolerance
    :param peak_scale_k: float, peak scale factor for intensity scaling (only for reference spectra), None for no scaling
    :param peak_intensity_power: float, peak intensity power
    :return: search engine object
    """
    prepare_ms2_lib(ms2db=msp_path,
                    mz_tol=mz_tol,
                    peak_scale_k=peak_scale_k,
                    peak_intensity_power=peak_intensity_power)

    return


if __name__ == "__main__":
    prepare_ms2db(msp_path='MSMS_library.msp', mz_tol=0.05, peak_scale_k=None, peak_intensity_power=0.5)
