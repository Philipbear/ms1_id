"""
A module to group metabolic features from unique compounds
1. annotate isotopes
2. annotate adducts
3. annotate in-source fragments
"""


import numpy as np
from collections import defaultdict


def annotate_isotope(d, mz_tol=0.015, rt_tol=0.1):
    """
    Function to annotate isotopes in a MS data.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object.
    """

    # rank the rois (d.rois) in each file by m/z
    d.rois.sort(key=lambda x: x.mz)

    for r in d.rois:

        if r.is_isotope:
            continue

        r.isotope_int_seq = [r.peak_height]
        r.isotope_mz_seq = [r.mz]

        last_mz = r.mz

        # check if the current ion is doubly charged
        r.charge_state = 1
        target_mz = r.mz + 1.003355 / 2
        v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]
        if len(v) > 0:
            # intensity rules against M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < 1.2 * r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > 0.01 * r.peak_height]
            if len(v) > 0:
                r.charge_state = 2

        isotope_id_seq = []
        target_mz = r.mz + 1.003355 / r.charge_state
        total_int = r.peak_height

        # find roi using isotope list
        i = 0
        while i < 5:  # maximum 5 isotopes
            # if isotope is not found in two daltons, stop searching
            if target_mz - last_mz > 2.2:
                break

            v = np.where(np.logical_and(np.abs(d.roi_mz_seq - target_mz) < mz_tol, np.abs(d.roi_rt_seq - r.rt) < rt_tol))[0]

            if len(v) == 0:
                i += 1
                continue

            # an isotope can't have intensity 1.2 fold or higher than M0 or 0.1% lower than the M0
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height < 1.2 * r.peak_height]
            v = [v[i] for i in range(len(v)) if d.rois[v[i]].peak_height > 0.001 * total_int]

            if len(v) == 0:
                i += 1
                continue

            # for high-resolution data, C and N isotopes can be separated and need to be summed
            total_int = np.sum([d.rois[vi].peak_height for vi in v])
            last_mz = np.mean([d.rois[vi].mz for vi in v])

            r.isotope_mz_seq.append(last_mz)
            r.isotope_int_seq.append(total_int)

            for vi in v:
                d.rois[vi].is_isotope = True
                isotope_id_seq.append(d.rois[vi].id)

            target_mz = last_mz + 1.003355 / r.charge_state
            i += 1

        r.charge_state = get_charge_state(r.isotope_mz_seq)
        r.isotope_id_seq = isotope_id_seq


def calc_all_ppc(d):
    """
    A function to calculate all peak-peak correlations

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.

    Returns
    ----------------------------------------------------------
    ppc_scores: dict
        A dictionary where keys are ROI IDs and values are dictionaries of
        {other_roi_id: ppc_score} pairs.
    """
    rois = d.rois  # Assuming d.rois is a list of Roi objects
    ppc_scores = defaultdict(dict)

    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois[i + 1:], start=i + 1):
            ppc = calculate_ppc(roi1, roi2)

            # Store the PPC score for both ROIs
            ppc_scores[roi1.id][roi2.id] = ppc
            ppc_scores[roi2.id][roi1.id] = ppc

    return ppc_scores


def annotate_adduct(d, mz_tol=0.01, rt_tol=0.05):
    """
    A function to annotate adducts from the same compound.

    Parameters
    ----------------------------------------------------------
    d: MSData object
        An MSData object that contains the detected rois to be grouped.
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

        _adduct_list = adduct_double_list if r.charge_state == 1 else adduct_single_list

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

            if calculate_ppc(r, d.rois[v]) > d.params.ppr:
                roi_to_label[v] = False
                d.rois[v].adduct_type = adduct[i]['name']
                d.rois[v].adduct_parent_roi_id = r.id
                r.adduct_child_roi_id.append(d.rois[v].id)

        if len(r.adduct_child_roi_id) > 0:
            r.adduct_type = default_single_adduct if r.charge_state == 1 else default_double_adduct

    for r in d.rois:
        if r.adduct_type is None:
            r.adduct_type = default_single_adduct if r.charge_state == 1 else default_double_adduct


def calculate_ppc(roi1, roi2):
    """
    A function to calculate the peak-peak correlation between two ROIs,
    filling in zeros for missing intensities and using the entire vector.

    Parameters
    ----------------------------------------------------------
    roi1: ROI object
        An ROI object with attributes scan_idx_seq and int_seq.
    roi2: ROI object
        An ROI object with attributes scan_idx_seq and int_seq.

    Returns
    ----------------------------------------------------------
    pp_cor: float
        The peak-peak correlation between the two ROIs.
    """
    # Find the union of all scan indices
    all_scans = np.union1d(roi1.scan_idx_seq, roi2.scan_idx_seq)

    # Create full intensity vectors for both ROIs
    int1 = np.zeros(len(all_scans))
    int2 = np.zeros(len(all_scans))

    # Fill in the intensities where they exist
    idx1 = np.searchsorted(all_scans, roi1.scan_idx_seq)
    idx2 = np.searchsorted(all_scans, roi2.scan_idx_seq)
    int1[idx1] = roi1.int_seq
    int2[idx2] = roi2.int_seq

    # Handle special cases
    if len(all_scans) <= 1:
        return 0.0  # Not enough data points for correlation
    if np.all(int1 == int1[0]) and np.all(int2 == int2[0]):
        return 1.0 if int1[0] == int2[0] else 0.0  # All values are the same

    # Calculate the correlation
    pp_cor = np.corrcoef(int1, int2)[0, 1]

    # Handle potential NaN result (e.g., if one vector is all zeros)
    return 0.0 if np.isnan(pp_cor) else pp_cor


def get_charge_state(mz_seq):
    if len(mz_seq) < 2:
        return 1
    else:
        mass_diff = mz_seq[1] - mz_seq[0]

        # check mass diff is closer to 1 or 0.5 | note, mass_diff can be larger than 1
        if abs(mass_diff - 1) < abs(mass_diff - 0.5):
            return 1
        else:
            return 2


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
        copy_1 = list(_adduct_pos)
        copy_2 = list(_adduct_pos)
        return copy_1.pop(0), copy_2.pop(16)
    else:
        copy_1 = list(_adduct_pos)
        copy_2 = list(_adduct_pos)
        return copy_1.pop(0), copy_2.pop(12)




_adduct_pos = [
    {'name': '[M+H]+', 'm': 1, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[M+Na]+', 'm': 1, 'charge': 1, 'mass': 22.989220702},
    {'name': '[M+K]+', 'm': 1, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[M+NH4]+', 'm': 1, 'charge': 1, 'mass': 18.03382555335},
    {'name': '[M-H+2Na]+', 'm': 1, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[M+H-H2O]+', 'm': 1, 'charge': 1, 'mass': -17.0032882318},
    {'name': '[M+H-2H2O]+', 'm': 1, 'charge': 1, 'mass': -35.01385291583},
    {'name': '[M+H-3H2O]+', 'm': 1, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[2M+H]+', 'm': 2, 'charge': 1, 'mass': 1.00727645223},
    {'name': '[2M+Na]+', 'm': 2, 'charge': 1, 'mass': 22.989220702},
    {'name': '[2M+K]+', 'm': 2, 'charge': 1, 'mass': 38.9631579064},
    {'name': '[2M+NH4]+', 'm': 2, 'charge': 1, 'mass': 18.03382555335},
    {'name': '[2M-H+2Na]+', 'm': 2, 'charge': 1, 'mass': 44.97116495177},
    {'name': '[2M+H-H2O]+', 'm': 2, 'charge': 1, 'mass': -17.0032882318},
    {'name': '[2M+H-2H2O]+', 'm': 2, 'charge': 1, 'mass': -35.01385291583},
    {'name': '[2M+H-3H2O]+', 'm': 2, 'charge': 1, 'mass': -53.02441759986},

    {'name': '[M+2H]2+', 'm': 1, 'charge': 2, 'mass': 2.01455290446},
    {'name': '[M+H+Na]2+', 'm': 1, 'charge': 2, 'mass': 23.99649715423},
    {'name': '[M+H+NH4]2+', 'm': 1, 'charge': 2, 'mass': 19.04110200558},
    {'name': '[M+Ca]2+', 'm': 1, 'charge': 2, 'mass': 39.961493703},
    {'name': '[M+Fe]2+', 'm': 1, 'charge': 2, 'mass': 55.93383917}
]

_adduct_neg = {
    {'name': '[M-H]-', 'm': 1, 'charge': 1, 'mass': -1.00727645223},
    {'name': '[M+Cl]-', 'm': 1, 'charge': 1, 'mass': 34.968304102},
    {'name': '[M+Br]-', 'm': 1, 'charge': 1, 'mass': 78.91778902},
    {'name': '[M+FA]-', 'm': 1, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[M+Ac]-', 'm': 1, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[M-H-H2O]-', 'm': 1, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[2M-H]-', 'm': 2, 'charge': 1, 'mass': -1.00727645223},
    {'name': '[2M+Cl]-', 'm': 2, 'charge': 1, 'mass': 34.968304102},
    {'name': '[2M+Br]-', 'm': 2, 'charge': 1, 'mass': 78.91778902},
    {'name': '[2M+FA]-', 'm': 2, 'charge': 1, 'mass': 44.99710569137},
    {'name': '[2M+Ac]-', 'm': 2, 'charge': 1, 'mass': 59.01275575583},
    {'name': '[2M-H-H2O]-', 'm': 2, 'charge': 1, 'mass': -19.01784113626},

    {'name': '[M-2H]2-', 'm': 1, 'charge': 2, 'mass': -2.01455290446},
    {'name': '[M-H+Cl]2-', 'm': 1, 'charge': 2, 'mass': 33.96157622977},
    {'name': '[M-H+Br]2-', 'm': 1, 'charge': 2, 'mass': 77.91106114777},
}
