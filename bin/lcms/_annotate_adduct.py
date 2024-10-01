"""
annotate adduct for single files and aligned features
"""

import numpy as np
from masscube.feature_grouping import peak_peak_correlation


def annotate_adduct(d, mz_tol=0.01, rt_tol=0.05):
    """
    A function to annotate adducts for single files

    Parameters
    ----------------------------------------------------------
    d: MSData object
    """

    d.rois.sort(key=lambda x: x.mz)
    roi_to_label = np.ones(len(d.rois), dtype=bool)

    # isotopes, in-source fragments
    for idx, r in enumerate(d.rois):
        if r.is_isotope or r.is_in_source_fragment:
            roi_to_label[idx] = False

    if d.params.ion_mode.lower() == "positive":
        default_single_adduct = "[M+H]+"
        default_double_adduct = "[M+2H]2+"
    elif d.params.ion_mode.lower() == "negative":
        default_single_adduct = "[M-H]-"
        default_double_adduct = "[M-2H]2-"

    adduct_single_list, adduct_double_list = _pop_adduct_list(_adduct_pos, _adduct_neg, d.params.ion_mode.lower())

    # find adducts by assuming the current roi is the [M+H]+ ion in positive mode and [M-H]- ion in negative mode
    for idx, r in enumerate(d.rois):

        if not roi_to_label[idx]:
            continue

        mol_mass = _calc_exact_mol_mass(r.mz, r.charge_state, d.params.ion_mode.lower())

        _adduct_list = adduct_single_list if r.charge_state == 1 else adduct_double_list

        for i, adduct in enumerate(_adduct_list):
            m = (mol_mass * _adduct_list[i]['m'] + _adduct_list[i]['mass']) / _adduct_list[i]['charge']
            v = np.logical_and(np.abs(d.roi_mz_seq - m) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol)
            v = np.where(np.logical_and(v, roi_to_label))[0]

            if len(v) == 0:
                continue

            if len(v) > 1:
                # select the one with the lowest RT difference
                v = v[np.argmin(np.abs(d.roi_rt_seq[v] - r.rt))]
            else:
                v = v[0]

            if peak_peak_correlation(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].adduct_type = adduct['name']
                d.rois[v].adduct_parent_roi_id = r.id
                r.adduct_child_roi_id.append(d.rois[v].id)

        if len(r.adduct_child_roi_id) > 0:
            roi_to_label[idx] = False
            r.adduct_type = default_single_adduct if r.charge_state == 1 else default_double_adduct

    for r in d.rois:
        if r.adduct_type is None:
            r.adduct_type = default_single_adduct if r.charge_state == 1 else default_double_adduct


def _calc_exact_mol_mass(mz, charge_state, ion_mode):
    if ion_mode == 'positive':
        if charge_state == 1:
            return mz - 1.00727645223
        else:
            return mz * 2.0 - 2.01455290446
    else:
        if charge_state == 1:
            return mz + 1.00727645223
        else:
            return mz * 2.0 + 2.01455290446


def _pop_adduct_list(_adduct_pos, _adduct_neg, ion_mode):
    if ion_mode == 'positive':
        return _adduct_pos[1:-5], _adduct_pos[-4:]  # Remove M+H, M+2H
    else:
        return _adduct_neg[1:-3], _adduct_neg[-2:]  # Remove M-H, M-2H


_adduct_pos = [
    {'name': '[M+H]+', 'm': 1, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[M+Na]+', 'm': 1, 'charge': 1, 'mass': 22.989220702},
    {'name': '[M+K]+', 'm': 1, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[M+NH4]+', 'm': 1, 'charge': 1, 'mass': 18.03382555335},
    # {'name': '[M-H+2Na]+', 'm': 1, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[M+H-H2O]+', 'm': 1, 'charge': 1, 'mass': -17.0032882318},
    {'name': '[M+H-2H2O]+', 'm': 1, 'charge': 1, 'mass': -35.01385291583},
    {'name': '[M+H-3H2O]+', 'm': 1, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[2M+H]+', 'm': 2, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[2M+Na]+', 'm': 2, 'charge': 1, 'mass': 22.989220702},
    {'name': '[2M+K]+', 'm': 2, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[2M+NH4]+', 'm': 2, 'charge': 1, 'mass': 18.03382555335},
    # {'name': '[2M-H+2Na]+', 'm': 2, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[2M+H-H2O]+', 'm': 2, 'charge': 1, 'mass': -17.0032882318},
    # {'name': '[2M+H-2H2O]+', 'm': 2, 'charge': 1, 'mass': -35.01385291583},
    # {'name': '[2M+H-3H2O]+', 'm': 2, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[M+2H]2+', 'm': 1, 'charge': 2, 'mass': 2.01455290446},
    # {'name': '[M+H+Na]2+', 'm': 1, 'charge': 2, 'mass': 23.99649715423},
    # {'name': '[M+H+NH4]2+', 'm': 1, 'charge': 2, 'mass': 19.04110200558},
    {'name': '[M+Ca]2+', 'm': 1, 'charge': 2, 'mass': 39.961493703},
    {'name': '[M+Fe]2+', 'm': 1, 'charge': 2, 'mass': 55.93383917}
]

_adduct_neg = [
    {'name': '[M-H]-', 'm': 1, 'charge': 1, 'mass': -1.00727645223},
    {'name': '[M+Cl]-', 'm': 1, 'charge': 1, 'mass': 34.968304102},
    {'name': '[M+Br]-', 'm': 1, 'charge': 1, 'mass': 78.91778902},
    {'name': '[M+FA]-', 'm': 1, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[M+Ac]-', 'm': 1, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[M-H-H2O]-', 'm': 1, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[2M-H]-', 'm': 2, 'charge': 1, 'mass': -1.00727645223},
    # {'name': '[2M+Cl]-', 'm': 2, 'charge': 1, 'mass': 34.968304102},
    # {'name': '[2M+Br]-', 'm': 2, 'charge': 1, 'mass': 78.91778902},
    {'name': '[2M+FA]-', 'm': 2, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[2M+Ac]-', 'm': 2, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[2M-H-H2O]-', 'm': 2, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[M-2H]2-', 'm': 1, 'charge': 2, 'mass': -2.01455290446},
    # {'name': '[M-H+Cl]2-', 'm': 1, 'charge': 2, 'mass': 33.96157622977},
    # {'name': '[M-H+Br]2-', 'm': 1, 'charge': 2, 'mass': 77.91106114777},
]
