from bin.lcms._reverse_matching import prepare_ms2_lib

######### prepare the search engine #########
prepare_ms2_lib(
    ms2db='data/gnps.msp',
    mz_tol=0.05,
    peak_scale_k=None,  # peak scale factor, None for no scaling
    peak_intensity_power=0.5  # peak intensity power, 0.5 for square root
)

prepare_ms2_lib(
    ms2db='data/gnps.msp',
    mz_tol=0.05,
    peak_scale_k=10,
    peak_intensity_power=0.5
)

######### load the search engine #########
# with open('data/gnps.pkl', 'rb') as file:
#     search_eng = pickle.load(file)
#
# print(search_eng)
