import pickle


def load_msms_db(path):
    """
    A function to load the MS/MS database in MSP format or pickle format.

    Parameters
    ----------
    path : str
        The path to the MS/MS database in MSP format.
    """

    print("Loading MS/MS database...")

    entropy_search = pickle.load(open(path, 'rb'))
    print("MS/MS database loaded.")
    return entropy_search
