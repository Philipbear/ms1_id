#!/usr/bin/env python3
import json
import multiprocessing
import pickle
from functools import reduce
from pathlib import Path
from typing import Union, List

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class FlashCosSearchCore:
    def __init__(self, path_data=None, max_ms2_tolerance_in_da=0.024, mz_index_step=0.0001, sqrt_transform=False):
        """
        Initialize the RevCos search core. for CPU only.
        :param path_array: The path array of the index files.
        :param max_ms2_tolerance_in_da: The maximum MS2 tolerance used when searching the MS/MS spectra, in Dalton. Default is 0.024.
        :param mz_index_step:   The step size of the m/z index, in Dalton. Default is 0.0001.
                                The smaller the step size, the faster the search, but the larger the index size and longer the index building time.
        """
        self.mz_index_step = mz_index_step
        self._init_for_multiprocessing = False
        self.max_ms2_tolerance_in_da = max_ms2_tolerance_in_da
        self.sqrt_transform = sqrt_transform

        self.total_spectra_num = 0
        self.total_peaks_num = 0
        self.index = []

        if path_data:
            self.path_data = Path(path_data)
        else:
            self.path_data = None

        self.index_names = [
            "all_ions_mz_idx_start",
            "all_ions_mz",
            "all_ions_intensity",
            "all_ions_spec_idx",
            "all_nl_mass_idx_start",
            "all_nl_mass",
            "all_nl_intensity",
            "all_nl_spec_idx",
            "all_ions_idx_for_nl",
        ]
        self.index_dtypes = {
            "all_ions_mz_idx_start": np.int64,
            "all_ions_mz": np.float32,
            "all_ions_intensity": np.float32,
            "all_ions_spec_idx": np.uint32,
            "all_nl_mass_idx_start": np.int64,
            "all_nl_mass": np.float32,
            "all_nl_intensity": np.float32,
            "all_nl_spec_idx": np.uint32,
            "all_ions_idx_for_nl": np.uint64,
        }

    def search(
            self,
            method="open",
            precursor_mz=None,
            peaks=None,
            ms2_tolerance_in_da=0.02,
            search_type=0,
            search_spectra_idx_min=0,
            search_spectra_idx_max=0
    ):
        """
        Perform identity-, open- or neutral loss search on the MS/MS spectra library.

        :param method:  The search method, can be "open" or "neutral_loss".
                        Set it to "open" for identity search and open search, set it to "neutral_loss" for neutral loss search.
        :param precursor_mz:    The precursor m/z of the query MS/MS spectrum, required for neutral loss search.
        :param peaks:   The peaks of the query MS/MS spectrum. The peaks need to be precleaned by "clean_spectrum" function.
        :param ms2_tolerance_in_da: The MS2 tolerance used when searching the MS/MS spectra, in Dalton. Default is 0.02.
        :param search_type: The search type, can be 0, 1 or 2.
                            Set it to 0 for searching the whole MS/MS spectra library.
                            Set it to 1 for searching a range of the MS/MS spectra library,
        :param search_spectra_idx_min:  The minimum index of the MS/MS spectra to search, required when search_type is 1.
        :param search_spectra_idx_max:  The maximum index of the MS/MS spectra to search, required when search_type is 1.
        """
        global library_mz
        if not self.index:
            return np.zeros(0, dtype=np.float32)
        if len(peaks) == 0:
            return (np.zeros(self.total_spectra_num, dtype=np.float32),
                    np.zeros(self.total_spectra_num, dtype=np.uint32),
                    np.zeros(self.total_spectra_num, dtype=np.float32))

        # Check peaks
        assert ms2_tolerance_in_da <= self.max_ms2_tolerance_in_da, "The MS2 tolerance is larger than the maximum MS2 tolerance."
        # assert abs(np.sum(np.square(peaks[:, 1])) - 1) < 1e-4, "The peaks are not normalized to sum to 1."
        assert (
                peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2
        ), "The peaks array should be sorted by m/z, and the m/z difference between two adjacent peaks should be larger than 2 * max_ms2_tolerance_in_da."
        (
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
            all_nl_mass_idx_start,
            all_nl_mass,
            all_nl_intensity,
            all_nl_spec_idx,
            all_ions_idx_for_nl,
        ) = self.index
        index_number_in_one_da = int(1 / self.mz_index_step)

        # Prepare the query spectrum
        peaks = self._preprocess_peaks(peaks)

        # Prepare the library
        if method == "open":
            library_mz_idx_start = all_ions_mz_idx_start
            library_mz = all_ions_mz
            library_peaks_intensity = all_ions_intensity
            library_spec_idx = all_ions_spec_idx
        else:  # neutral loss
            library_mz_idx_start = all_nl_mass_idx_start
            library_mz = all_nl_mass
            library_peaks_intensity = all_nl_intensity
            library_spec_idx = all_nl_spec_idx
            peaks[:, 0] = precursor_mz - peaks[:, 0]

        # Start searching
        # Calculate the cosine similarity for this matched peak
        cos_similarity = np.zeros(self.total_spectra_num, dtype=np.float32)
        cos_matched_cnt = np.zeros(self.total_spectra_num, dtype=np.uint32)
        # sum(matched peaks intensity) / sum(query peaks intensity)
        cos_spectral_usage = np.zeros(self.total_spectra_num, dtype=np.float32)

        # a large empty 2D array to record matched peak pairs in query spectrum
        match_table_q = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)
        # a large empty 2D array to record matched peak pairs in ref. spectrum
        match_table_r = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)

        # loop through all peaks in query spectrum
        for k, (mz_query, intensity_query) in enumerate(peaks):
            # Determine the mz index range
            product_mz_idx_min = self._find_location_from_array_with_index(
                mz_query - ms2_tolerance_in_da, library_mz, library_mz_idx_start, "left", index_number_in_one_da
            )
            product_mz_idx_max = self._find_location_from_array_with_index(
                mz_query + ms2_tolerance_in_da, library_mz, library_mz_idx_start, "right", index_number_in_one_da
            )

            # intensity of matched peaks
            intensity_library = library_peaks_intensity[product_mz_idx_min:product_mz_idx_max]
            # library IDs of matched peaks
            modified_idx = library_spec_idx[product_mz_idx_min:product_mz_idx_max]

            # fill in the match table
            match_table_q[k, modified_idx] = intensity_query
            match_table_r[k, modified_idx] = intensity_library

        # search type 0: search the whole library
        if search_type == 0:
            # calculate spectral usage
            cos_spectral_usage = np.sum(match_table_q, axis=0) / np.sum(peaks[:, 1])
            # normalize query spectrum intensity in each column
            if self.sqrt_transform:
                match_table_q = np.sqrt(match_table_q)
                peaks[:, 1] = np.sqrt(peaks[:, 1])

            match_table_q = match_table_q / np.sqrt(np.sum(np.square(peaks[:, 1]), axis=0))

            # calculate cosine similarity
            cos_similarity = np.sum(match_table_q * match_table_r, axis=0)
            # count matched peaks
            cos_matched_cnt = np.sum(match_table_q > 0, axis=0)
        elif search_type == 1:
            # search type 1: search a range of the library, for identity search
            match_table_q = match_table_q[:, search_spectra_idx_min:search_spectra_idx_max]
            match_table_r = match_table_r[:, search_spectra_idx_min:search_spectra_idx_max]
            # calculate spectral usage
            part_cos_spectral_usage = np.sum(match_table_q, axis=0) / np.sum(peaks[:, 1])
            # normalize query spectrum intensity in each column
            if self.sqrt_transform:
                match_table_q = np.sqrt(match_table_q)
                peaks[:, 1] = np.sqrt(peaks[:, 1])

            match_table_q = match_table_q / np.sqrt(np.sum(np.square(peaks[:, 1])))

            # calculate reverse cosine similarity
            part_cos_similarity = np.sum(match_table_q * match_table_r, axis=0)
            # count matched peaks
            part_cos_matched_cnt = np.sum(match_table_q > 0, axis=0)

            # update the similarity and matched count
            cos_similarity[search_spectra_idx_min:search_spectra_idx_max] = part_cos_similarity
            cos_matched_cnt[search_spectra_idx_min:search_spectra_idx_max] = part_cos_matched_cnt
            cos_spectral_usage[search_spectra_idx_min:search_spectra_idx_max] = part_cos_spectral_usage

        del match_table_q, match_table_r

        return cos_similarity, cos_matched_cnt, cos_spectral_usage

    def search_hybrid(self, precursor_mz=None, peaks=None, ms2_tolerance_in_da=0.02):
        """
        Perform the hybrid search for the MS/MS spectra.

        :param precursor_mz: The precursor m/z of the MS/MS spectra.
        :param peaks: The peaks of the MS/MS spectra, needs to be cleaned with the "clean_spectrum" function.
        :param ms2_tolerance_in_da: The MS/MS tolerance in Da.
        """
        if not self.index:
            return np.zeros(0, dtype=np.float32)
        if len(peaks) == 0:
            return np.zeros(self.total_spectra_num, dtype=np.float32), np.zeros(self.total_spectra_num,
                                                                                dtype=np.uint32), np.zeros(
                self.total_spectra_num, dtype=np.float32)

        # Check peaks
        assert ms2_tolerance_in_da <= self.max_ms2_tolerance_in_da, "The MS2 tolerance is larger than the maximum MS2 tolerance."
        # assert abs(np.sum(np.square(peaks[:, 1])) - 1) < 1e-4, "The peaks are not normalized to sum to 1."
        assert (
                peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2
        ), "The peaks array should be sorted by m/z, and the m/z difference between two adjacent peaks should be larger than 2 * max_ms2_tolerance_in_da."
        (
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
            all_nl_mass_idx_start,
            all_nl_mass,
            all_nl_intensity,
            all_nl_spec_idx,
            all_ions_idx_for_nl,
        ) = self.index
        index_number_in_one_da = int(1 / self.mz_index_step)

        # Prepare the query spectrum
        peaks = self._preprocess_peaks(peaks)

        # Go through all peak in the spectrum and determine the mz index range
        product_peak_match_idx_min = np.zeros(peaks.shape[0], dtype=np.uint64)
        product_peak_match_idx_max = np.zeros(peaks.shape[0], dtype=np.uint64)
        for peak_idx, (mz_query, _) in enumerate(peaks):
            # Determine the mz index range
            product_mz_idx_min = self._find_location_from_array_with_index(
                mz_query - ms2_tolerance_in_da, all_ions_mz, all_ions_mz_idx_start, "left", index_number_in_one_da
            )
            product_mz_idx_max = self._find_location_from_array_with_index(
                mz_query + ms2_tolerance_in_da, all_ions_mz, all_ions_mz_idx_start, "right", index_number_in_one_da
            )

            product_peak_match_idx_min[peak_idx] = product_mz_idx_min
            product_peak_match_idx_max[peak_idx] = product_mz_idx_max

        # a large empty 2D array to record matched peak pairs in query spectrum
        match_table_q = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)
        match_table_q_nl = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)  # for neutral loss
        # a large empty 2D array to record matched peak pairs in ref. spectrum
        match_table_r = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)
        match_table_r_nl = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)  # for neutral loss

        # Go through all the peaks in the spectrum and calculate the cosine similarity
        for k, (mz, intensity) in enumerate(peaks):
            ###############################################################
            # Match the original product ion
            product_mz_idx_min = product_peak_match_idx_min[k]
            product_mz_idx_max = product_peak_match_idx_max[k]

            # intensity of matched peaks
            intensity_library = all_ions_intensity[product_mz_idx_min:product_mz_idx_max]
            # library IDs of matched peaks
            modified_idx = all_ions_spec_idx[product_mz_idx_min:product_mz_idx_max]

            # fill in the match table
            match_table_q[k, modified_idx] = intensity
            match_table_r[k, modified_idx] = intensity_library

            ###############################################################
            # Match the neutral loss ions
            mz_nl = precursor_mz - mz
            # Determine the mz index range
            neutral_loss_mz_idx_min = self._find_location_from_array_with_index(
                mz_nl - ms2_tolerance_in_da, all_nl_mass, all_nl_mass_idx_start, "left", index_number_in_one_da
            )
            neutral_loss_mz_idx_max = self._find_location_from_array_with_index(
                mz_nl + ms2_tolerance_in_da, all_nl_mass, all_nl_mass_idx_start, "right", index_number_in_one_da
            )

            # intensity of matched peaks
            intensity_library_nl = all_nl_intensity[neutral_loss_mz_idx_min:neutral_loss_mz_idx_max].copy()
            modified_idx_nl = all_nl_spec_idx[neutral_loss_mz_idx_min:neutral_loss_mz_idx_max]

            # Check if the neutral loss ion is already matched to other query peak as a product ion
            nl_matched_product_ion_idx = all_ions_idx_for_nl[neutral_loss_mz_idx_min:neutral_loss_mz_idx_max]
            s1 = np.searchsorted(product_peak_match_idx_min, nl_matched_product_ion_idx, side="right")
            s2 = np.searchsorted(product_peak_match_idx_max - 1, nl_matched_product_ion_idx, side="left")
            intensity_library_nl[s1 > s2] = 0

            # Check if this query peak is already matched to a product ion in the same library spectrum
            duplicate_idx_in_nl = self._remove_duplicate_with_cpu(modified_idx, modified_idx_nl,
                                                                  self.total_spectra_num)
            intensity_library_nl[duplicate_idx_in_nl] = 0

            # fill in the match table for neutral loss
            match_table_q_nl[k, modified_idx_nl] = intensity
            match_table_r_nl[k, modified_idx_nl] = intensity_library_nl

        # merge the match table
        match_table_q_all = match_table_q + match_table_q_nl
        match_table_r_all = match_table_r + match_table_r_nl
        match_table_q_max = np.maximum(match_table_q, match_table_q_nl)

        del match_table_q, match_table_r, match_table_q_nl, match_table_r_nl

        # calculate spectral usage
        cos_spectral_usage = np.sum(match_table_q_max, axis=0) / np.sum(peaks[:, 1])
        # normalize query spectrum intensity in each column
        if self.sqrt_transform:
            match_table_q_all = np.sqrt(match_table_q_all)
            peaks[:, 1] = np.sqrt(peaks[:, 1])

        match_table_q_all = match_table_q_all / np.sqrt(np.sum(np.square(peaks[:, 1]), axis=0))
        # calculate cosine similarity
        cos_similarity = np.sum(match_table_q_all * match_table_r_all, axis=0)
        # count matched peaks
        cos_matched_cnt = np.sum(match_table_q_all > 0, axis=0)
        del match_table_q_all, match_table_r_all

        return cos_similarity, cos_matched_cnt, cos_spectral_usage

    def search_mass(self, query_mass, ms2_tolerance_in_da=0.02, method="open"):
        """
        Search a specific fragment m/z or neutral loss in the MS/MS spectra library.
        :param query_mass: The fragment m/z or neutral loss to search.
        :param ms2_tolerance_in_da: The tolerance used when searching the MS/MS spectra, in Dalton. Default is 0.02.
        :param method: open or neutral_loss
        :return: The matched peak intensity in the library.
        """
        global library_mz
        (
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
            all_nl_mass_idx_start,
            all_nl_mass,
            all_nl_intensity,
            all_nl_spec_idx,
            all_ions_idx_for_nl,
        ) = self.index
        index_number_in_one_da = int(1 / self.mz_index_step)

        # Prepare the library
        if method == "open":
            library_mz_idx_start = all_ions_mz_idx_start
            library_mz = all_ions_mz
            library_peaks_intensity = all_ions_intensity
            library_spec_idx = all_ions_spec_idx
        else:  # neutral loss
            library_mz_idx_start = all_nl_mass_idx_start
            library_mz = all_nl_mass
            library_peaks_intensity = all_nl_intensity
            library_spec_idx = all_nl_spec_idx

        # Start searching
        # matched peak intensity in the library
        matched_peak_intensity = np.zeros(self.total_spectra_num, dtype=np.float32)

        # loop through all peaks in query spectrum
        # Determine the mz index range
        product_mz_idx_min = self._find_location_from_array_with_index(
            query_mass - ms2_tolerance_in_da, library_mz, library_mz_idx_start, "left", index_number_in_one_da
        )
        product_mz_idx_max = self._find_location_from_array_with_index(
            query_mass + ms2_tolerance_in_da, library_mz, library_mz_idx_start, "right", index_number_in_one_da
        )

        # intensity of matched peaks
        intensity_library = library_peaks_intensity[product_mz_idx_min:product_mz_idx_max]
        # library IDs of matched peaks
        modified_idx = library_spec_idx[product_mz_idx_min:product_mz_idx_max]
        # fill in the match table
        matched_peak_intensity[modified_idx] = intensity_library

        return matched_peak_intensity

    def _remove_duplicate_with_cpu(self, array_1, array_2, max_element):
        if len(array_1) + len(array_2) < 4_000_000:
            # When len(array_1) + len(array_2) < 4_000_000, this method is faster than array method
            aux = np.concatenate((array_1, array_2))
            aux_sort_indices = np.argsort(aux, kind="mergesort")
            aux = aux[aux_sort_indices]
            mask = aux[1:] == aux[:-1]
            array2_indices = aux_sort_indices[1:][mask] - array_1.size
            return array2_indices
        else:
            # When len(array_1) + len(array_2) > 4_000_000, this method is faster than sort method
            note = np.zeros(max_element, dtype=np.int8)
            note[array_1] = 1
            duplicate_idx = np.where(note[array_2] == 1)[0]
            return duplicate_idx

    def build_index(self, all_spectra_list: list, max_indexed_mz: float = 1500.00005):
        """
        Build the index for the MS/MS spectra library.

        The spectra provided to this function should be a dictionary in the format of {"precursor_mz": precursor_mz, "peaks": peaks}.
        The precursor_mz is the precursor m/z value of the MS/MS spectrum;
        The peaks is a numpy array which has been processed by the function "clean_spectrum".

        :param all_spectra_list:    A list of dictionaries in the format of {"precursor_mz": precursor_mz, "peaks": peaks},
                                    the spectra in the list need to be sorted by the precursor m/z.
        :param max_indexed_mz: The maximum m/z value that will be indexed. Default is 1500.00005.
        """

        # Get the total number of spectra and peaks
        total_peaks_num = np.sum([spectrum["peaks"].shape[0] for spectrum in all_spectra_list])
        total_spectra_num = len(all_spectra_list)
        # total_spectra_num can not be bigger than 2^32-1 (uint32), total_peak_num can not be bigger than 2^63-1 (int64)
        assert total_spectra_num < 2 ** 32 - 1, "The total spectra number is too big."
        assert total_peaks_num < 2 ** 63 - 1, "The total peaks number is too big."
        self.total_spectra_num = total_spectra_num
        self.total_peaks_num = total_peaks_num

        ############## Step 1: Collect the precursor m/z and peaks information. ##############
        dtype_peak_data = np.dtype(
            [
                ("ion_mz", np.float32),  # The m/z of the fragment ion.
                ("nl_mass", np.float32),  # The neutral loss mass of the fragment ion.
                ("intensity", np.float32),  # The intensity of the fragment ion.
                ("spec_idx", np.uint32),  # The index of the MS/MS spectra.
                ("peak_idx", np.uint64),
            ],
            align=True,
        )  # The index of the fragment ion.

        # Initialize the peak data array.
        peak_data = np.zeros(total_peaks_num, dtype=dtype_peak_data)
        peak_idx = 0

        # Adding the precursor m/z and peaks information to the peak data array.
        for idx, spectrum in enumerate(all_spectra_list):
            precursor_mz, peaks = spectrum["precursor_mz"], spectrum["peaks"]
            # Check the peaks array.
            assert peaks.ndim == 2, "The peaks array should be a 2D numpy array."
            assert peaks.shape[1] == 2, "The peaks array should be a 2D numpy array with the shape of [n, 2]."
            assert peaks.shape[0] > 0, "The peaks array should not be empty."
            assert abs(np.sum(np.square(peaks[:, 1])) - 1) < 1e-4, "The peaks array should be normalized to sum to 1."
            assert (
                    peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2
            ), "The peaks array should be sorted by m/z, and the m/z difference between two adjacent peaks should be larger than 2 * max_ms2_tolerance_in_da."

            # Preprocess the peaks array.
            peaks = self._preprocess_peaks(peaks)

            # Assign the product ion m/z
            peak_data_item = peak_data[peak_idx: (peak_idx + peaks.shape[0])]
            peak_data_item["ion_mz"] = peaks[:, 0]
            # Assign the neutral loss mass
            peak_data_item["nl_mass"] = precursor_mz - peaks[:, 0]
            # Assign the intensity
            peak_data_item["intensity"] = peaks[:, 1]
            # Assign the spectrum index
            peak_data_item["spec_idx"] = idx
            # Set the peak index
            peak_idx += peaks.shape[0]

        ############## Step 2: Build the index by sort with product ions. ##############
        self.index = self._generate_index_from_peak_data(peak_data, max_indexed_mz)
        return self.index

    def _generate_index_from_peak_data(self, peak_data, max_indexed_mz):
        # Sort with precursor m/z.
        peak_data.sort(order="ion_mz")

        # Record the m/z, intensity, and spectrum index information for product ions.
        all_ions_mz = np.copy(peak_data["ion_mz"])
        all_ions_intensity = np.copy(peak_data["intensity"])
        all_ions_spec_idx = np.copy(peak_data["spec_idx"])

        # Assign the index of the product ions.
        peak_data["peak_idx"] = np.arange(0, self.total_peaks_num, dtype=np.uint64)

        # Build index for fast access to the ion's m/z.
        max_mz = min(np.max(all_ions_mz), max_indexed_mz)
        search_array = np.arange(0.0, max_mz, self.mz_index_step)
        all_ions_mz_idx_start = np.searchsorted(all_ions_mz, search_array, side="left").astype(np.int64)

        ############## Step 3: Build the index by sort with neutral loss mass. ##############
        # Sort with the neutral loss mass.
        peak_data.sort(order="nl_mass")

        # Record the m/z, intensity, spectrum index, and product ions index information for neutral loss ions.
        all_nl_mass = peak_data["nl_mass"]
        all_nl_intensity = peak_data["intensity"]
        all_nl_spec_idx = peak_data["spec_idx"]
        all_ions_idx_for_nl = peak_data["peak_idx"]

        # Build the index for fast access to the neutral loss mass.
        max_mz = min(np.max(all_nl_mass), max_indexed_mz)
        search_array = np.arange(0.0, max_mz, self.mz_index_step)
        all_nl_mass_idx_start = np.searchsorted(all_nl_mass, search_array, side="left").astype(np.int64)

        ############## Step 4: Save the index. ##############
        index = [
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
            all_nl_mass_idx_start,
            all_nl_mass,
            all_nl_intensity,
            all_nl_spec_idx,
            all_ions_idx_for_nl,
        ]
        return index

    def _preprocess_peaks(self, peaks):
        """
        Preprocess the peaks.
        """
        peaks_clean = np.asarray(peaks).copy()
        return peaks_clean

    def _find_location_from_array_with_index(self, wanted_mz, mz_array, mz_idx_start_array, side, index_number):
        mz_min_int = (np.floor(wanted_mz * index_number)).astype(int)
        mz_max_int = mz_min_int + 1

        if mz_min_int >= len(mz_idx_start_array):
            mz_idx_search_start = mz_idx_start_array[-1]
        else:
            mz_idx_search_start = mz_idx_start_array[mz_min_int].astype(int)

        if mz_max_int >= len(mz_idx_start_array):
            mz_idx_search_end = len(mz_array)
        else:
            mz_idx_search_end = mz_idx_start_array[mz_max_int].astype(int) + 1

        return mz_idx_search_start + np.searchsorted(mz_array[mz_idx_search_start:mz_idx_search_end], wanted_mz,
                                                     side=side)

    def save_memory_for_multiprocessing(self):
        """
        Move the numpy array in the index to shared memory in order to save memory.
        This function is not required when you only use one thread to search the MS/MS spectra.
        When use multiple threads, this function is also not required but highly recommended, as it avoids the memory copy and saves a lot of memory and time.
        """
        if self._init_for_multiprocessing:
            return

        for i, array in enumerate(self.index):
            self.index[i] = _convert_numpy_array_to_shared_memory(array)
        self._init_for_multiprocessing = True

    def read(self, path_data=None):
        """
        Read the index from the specified path.
        """
        try:
            if path_data is None:
                path_data = self.path_data

            path_data = Path(path_data)
            self.index = []
            for name in self.index_names:
                self.index.append(np.fromfile(path_data / f"{name}.npy", dtype=self.index_dtypes[name]))

            with open(path_data / "information.json", "r") as f:
                information = json.load(f)
            self.mz_index_step = information["mz_index_step"]
            self.total_spectra_num = information["total_spectra_num"]
            self.total_peaks_num = information["total_peaks_num"]
            self.max_ms2_tolerance_in_da = information["max_ms2_tolerance_in_da"]
            return True
        except:
            return False

    def write(self, path_data=None):
        """
        Write the index to the specified path.
        """
        if path_data is None:
            path_data = self.path_data

        path_data = Path(path_data)
        path_data.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(self.index_names):
            self.index[i].tofile(str(path_data / f"{name}.npy"))
        information = {
            "mz_index_step": float(self.mz_index_step),
            "total_spectra_num": int(self.total_spectra_num),
            "total_peaks_num": int(self.total_peaks_num),
            "max_ms2_tolerance_in_da": float(self.max_ms2_tolerance_in_da),
        }
        with open(path_data / "information.json", "w") as f:
            json.dump(information, f)


def _convert_numpy_array_to_shared_memory(np_array, array_c_type=None):
    """
    The char table of shared memory can be find at:
    https://docs.python.org/3/library/struct.html#format-characters
    https://docs.python.org/3/library/array.html#module-array (This one is wrong!)
    The documentation of numpy.frombuffer can be find at:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html
    Note: the char table is different from the char table in numpy
    """
    dim = np_array.shape
    num = reduce(lambda x, y: x * y, dim)
    if array_c_type is None:
        array_c_type = np_array.dtype.char
    base = multiprocessing.Array(array_c_type, num, lock=False)
    np_array_new = np.frombuffer(base, dtype=np_array.dtype).reshape(dim)
    np_array_new[:] = np_array
    return np_array_new


class FlashCosSearchCoreLowMemory(FlashCosSearchCore):
    def __init__(self, path_data, max_ms2_tolerance_in_da=0.024, mz_index_step=0.0001, sqrt_transform=False) -> None:
        super().__init__(max_ms2_tolerance_in_da=max_ms2_tolerance_in_da,
                         mz_index_step=mz_index_step, sqrt_transform=sqrt_transform)
        self.path_data = Path(str(path_data))
        self.path_data.mkdir(parents=True, exist_ok=True)
        self.index_file = []

    def _generate_index_from_peak_data(self, peak_data, max_indexed_mz):
        total_peaks_num = peak_data.shape[0]

        # Sort with precursor m/z.
        peak_data.sort(order="ion_mz")

        # Record the m/z, intensity, and spectrum index information for product ions.
        (peak_data["ion_mz"]).tofile(self.path_data / "all_ions_mz.npy")
        (peak_data["intensity"]).tofile(self.path_data / "all_ions_intensity.npy")
        (peak_data["spec_idx"]).tofile(self.path_data / "all_ions_spec_idx.npy")

        # all_ions_mz = self._convert_view_to_array(peak_data.view(np.float32).reshape(total_peaks_num, -1)[:, 0], np.float32, "all_ions_mz")
        # all_ions_intensity = self._convert_view_to_array(peak_data.view(np.float32).reshape(total_peaks_num, -1)[:, 2], np.float32, "all_ions_intensity")
        # all_ions_spec_idx = self._convert_view_to_array(peak_data.view(np.uint32).reshape(total_peaks_num, -1)[:, 3], np.uint32, "all_ions_spec_idx")

        # Assign the index of the product ions.
        peak_data["peak_idx"] = np.arange(0, self.total_peaks_num, dtype=np.uint64)

        # Build index for fast access to the ion's m/z.
        all_ions_mz = np.memmap(self.path_data / "all_ions_mz.npy", dtype=np.float32, mode="r",
                                shape=(total_peaks_num,))
        max_mz = min(np.max(all_ions_mz), max_indexed_mz)
        search_array = np.arange(0.0, max_mz, self.mz_index_step)
        all_ions_mz_idx_start = np.searchsorted(all_ions_mz, search_array, side="left").astype(np.int64)
        all_ions_mz_idx_start.tofile(self.path_data / "all_ions_mz_idx_start.npy")

        ############## Step 3: Build the index by sort with neutral loss mass. ##############
        # Sort with the neutral loss mass.
        peak_data.sort(order="nl_mass")

        # Record the m/z, intensity, spectrum index, and product ions index information for neutral loss ions.
        (peak_data["nl_mass"]).tofile(self.path_data / "all_nl_mass.npy")
        (peak_data["intensity"]).tofile(self.path_data / "all_nl_intensity.npy")
        (peak_data["spec_idx"]).tofile(self.path_data / "all_nl_spec_idx.npy")
        (peak_data["peak_idx"]).tofile(self.path_data / "all_ions_idx_for_nl.npy")

        # all_nl_mass = self._convert_view_to_array(peak_data.view(np.float32).reshape(total_peaks_num, -1)[:, 1], np.float32, "all_nl_mass")
        # all_nl_intensity = self._convert_view_to_array(peak_data.view(np.float32).reshape(total_peaks_num, -1)[:, 2], np.float32, "all_nl_intensity")
        # all_nl_spec_idx = self._convert_view_to_array(peak_data.view(np.uint32).reshape(total_peaks_num, -1)[:, 3], np.uint32, "all_nl_spec_idx")
        # all_ions_idx_for_nl = self._convert_view_to_array(peak_data.view(np.uint64).reshape(total_peaks_num, -1)[:, 2], np.uint64, "all_ions_idx_for_nl")

        # Build the index for fast access to the neutral loss mass.
        all_nl_mass = np.memmap(self.path_data / "all_nl_mass.npy", dtype=np.float32, mode="r",
                                shape=(total_peaks_num,))
        max_mz = min(np.max(all_nl_mass), max_indexed_mz)
        search_array = np.arange(0.0, max_mz, self.mz_index_step)
        all_nl_mass_idx_start = np.searchsorted(all_nl_mass, search_array, side="left").astype(np.int64)
        all_nl_mass_idx_start.tofile(self.path_data / "all_nl_mass_idx_start.npy")

        ############## Step 4: Save the index. ##############
        self.write()
        self.read()
        return self.index

    def read(self, path_data=None):
        """
        Read the index from the file.
        """
        if path_data is not None:
            self.path_data = Path(path_data)

        try:
            self.index = []
            for name in self.index_names:
                self.index.append(np.memmap(self.path_data / f"{name}.npy", dtype=self.index_dtypes[name], mode="r"))
                file_cur = open(self.path_data / f"{name}.npy", "rb")
                file_cur.data_type = np.dtype(self.index_dtypes[name])
                self.index_file.append(file_cur)

            information = json.load(open(self.path_data / "information.json", "r"))
            self.mz_index_step = information["mz_index_step"]
            self.total_spectra_num = information["total_spectra_num"]
            self.total_peaks_num = information["total_peaks_num"]
            self.max_ms2_tolerance_in_da = information["max_ms2_tolerance_in_da"]
            return True
        except:
            return False

    def write(self, path_data=None):
        """
        Write the index to the file.
        """
        if path_data is not None:
            assert Path(path_data) == self.path_data, "The path_data is not the same as the path_data in the class."

        information = {
            "mz_index_step": float(self.mz_index_step),
            "total_spectra_num": int(self.total_spectra_num),
            "total_peaks_num": int(self.total_peaks_num),
            "max_ms2_tolerance_in_da": float(self.max_ms2_tolerance_in_da),
        }
        json.dump(information, open(self.path_data / "information.json", "w"))

    def search_hybrid(self, precursor_mz=None, peaks=None, ms2_tolerance_in_da=0.02):
        """
        Perform the hybrid search for the MS/MS spectra.

        :param precursor_mz: The precursor m/z of the MS/MS spectra.
        :param peaks: The peaks of the MS/MS spectra, needs to be cleaned with the "clean_spectrum" function.
        :param ms2_tolerance_in_da: The MS/MS tolerance in Da.
        """
        if not self.index:
            return np.zeros(0, dtype=np.float32)
        if len(peaks) == 0:
            return (np.zeros(self.total_spectra_num, dtype=np.float32),
                    np.zeros(self.total_spectra_num, dtype=np.float32),
                    np.zeros(self.total_spectra_num, dtype=np.float32))

        # Check peaks
        assert ms2_tolerance_in_da <= self.max_ms2_tolerance_in_da, "The MS2 tolerance is larger than the maximum MS2 tolerance."
        # assert abs(np.sum(np.square(peaks[:, 1])) - 1) < 1e-4, "The peaks are not normalized to sum to 1."
        assert (
                peaks.shape[0] <= 1 or np.min(peaks[1:, 0] - peaks[:-1, 0]) > self.max_ms2_tolerance_in_da * 2
        ), "The peaks array should be sorted by m/z, and the m/z difference between two adjacent peaks should be larger than 2 * max_ms2_tolerance_in_da."
        (
            all_ions_mz_idx_start,
            all_ions_mz,
            all_ions_intensity,
            all_ions_spec_idx,
            all_nl_mass_idx_start,
            all_nl_mass,
            all_nl_intensity,
            all_nl_spec_idx,
            all_ions_idx_for_nl,
        ) = self.index
        (
            file_all_ions_mz_idx_start,
            file_all_ions_mz,
            file_all_ions_intensity,
            file_all_ions_spec_idx,
            file_all_nl_mass_idx_start,
            file_all_nl_mass,
            file_all_nl_intensity,
            file_all_nl_spec_idx,
            file_all_ions_idx_for_nl,
        ) = self.index_file
        index_number_in_one_da = int(1 / self.mz_index_step)

        # Prepare the query spectrum
        peaks = self._preprocess_peaks(peaks)

        # Go through all peak in the spectrum and determine the mz index range
        product_peak_match_idx_min = np.zeros(peaks.shape[0], dtype=np.uint64)
        product_peak_match_idx_max = np.zeros(peaks.shape[0], dtype=np.uint64)
        for peak_idx, (mz_query, _) in enumerate(peaks):
            # Determine the mz index range
            product_mz_idx_min = self._find_location_from_array_with_index(
                mz_query - ms2_tolerance_in_da, all_ions_mz, all_ions_mz_idx_start, "left", index_number_in_one_da
            )
            product_mz_idx_max = self._find_location_from_array_with_index(
                mz_query + ms2_tolerance_in_da, all_ions_mz, all_ions_mz_idx_start, "right", index_number_in_one_da
            )

            product_peak_match_idx_min[peak_idx] = product_mz_idx_min
            product_peak_match_idx_max[peak_idx] = product_mz_idx_max

        # a large empty 2D array to record matched peak pairs in query spectrum
        match_table_q = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)
        match_table_q_nl = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)  # for neutral loss
        # a large empty 2D array to record matched peak pairs in ref. spectrum
        match_table_r = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)
        match_table_r_nl = np.zeros((peaks.shape[0], self.total_spectra_num), dtype=np.float32)  # for neutral loss

        # Go through all the peaks in the spectrum and calculate the cosine similarity
        for k, (mz, intensity) in enumerate(peaks):
            ###############################################################
            # Match the original product ion
            product_mz_idx_min = product_peak_match_idx_min[k]
            product_mz_idx_max = product_peak_match_idx_max[k]

            # intensity of matched peaks
            intensity_library = _read_data_from_file(file_all_ions_intensity, product_mz_idx_min,
                                                     product_mz_idx_max)
            # library IDs of matched peaks
            modified_idx = _read_data_from_file(file_all_ions_spec_idx, product_mz_idx_min,
                                                product_mz_idx_max)

            # fill in the match table
            match_table_q[k, modified_idx] = intensity
            match_table_r[k, modified_idx] = intensity_library

            ###############################################################
            # Match the neutral loss ions
            mz_nl = precursor_mz - mz
            # Determine the mz index range
            neutral_loss_mz_idx_min = self._find_location_from_array_with_index(
                mz_nl - ms2_tolerance_in_da, all_nl_mass, all_nl_mass_idx_start, "left", index_number_in_one_da
            )
            neutral_loss_mz_idx_max = self._find_location_from_array_with_index(
                mz_nl + ms2_tolerance_in_da, all_nl_mass, all_nl_mass_idx_start, "right", index_number_in_one_da
            )

            # intensity of matched peaks
            intensity_library_nl = _read_data_from_file(file_all_nl_intensity, neutral_loss_mz_idx_min,
                                                        neutral_loss_mz_idx_max).copy()
            modified_idx_nl = _read_data_from_file(file_all_nl_spec_idx, neutral_loss_mz_idx_min,
                                                   neutral_loss_mz_idx_max)

            # Check if the neutral loss ion is already matched to other query peak as a product ion
            nl_matched_product_ion_idx = all_ions_idx_for_nl[neutral_loss_mz_idx_min:neutral_loss_mz_idx_max]
            s1 = np.searchsorted(product_peak_match_idx_min, nl_matched_product_ion_idx, side="right")
            s2 = np.searchsorted(product_peak_match_idx_max - 1, nl_matched_product_ion_idx, side="left")
            intensity_library_nl[s1 > s2] = 0

            # Check if this query peak is already matched to a product ion in the same library spectrum
            duplicate_idx_in_nl = self._remove_duplicate_with_cpu(modified_idx, modified_idx_nl,
                                                                  self.total_spectra_num)
            intensity_library_nl[duplicate_idx_in_nl] = 0

            # fill in the match table for neutral loss
            match_table_q_nl[k, modified_idx_nl] = intensity
            match_table_r_nl[k, modified_idx_nl] = intensity_library_nl

        # merge the match table
        match_table_q_all = match_table_q + match_table_q_nl
        match_table_r_all = match_table_r + match_table_r_nl
        match_table_q_max = np.maximum(match_table_q, match_table_q_nl)

        del match_table_q, match_table_r, match_table_q_nl, match_table_r_nl

        # calculate spectral usage
        cos_spectral_usage = np.sum(match_table_q_max, axis=0) / np.sum(peaks[:, 1])
        # normalize query spectrum intensity in each column
        if self.sqrt_transform:
            match_table_q_all = np.sqrt(match_table_q_all)
            peaks[:, 1] = np.sqrt(peaks[:, 1])
        match_table_q_all = match_table_q_all / np.sqrt(np.sum(np.square(peaks[:, 1]), axis=0))
        # calculate reverse cosine similarity
        cos_similarity = np.sum(match_table_q_all * match_table_r_all, axis=0)
        # count matched peaks
        cos_matched_cnt = np.sum(match_table_q_all > 0, axis=0)
        del match_table_q_all, match_table_r_all

        return cos_similarity, cos_matched_cnt, cos_spectral_usage


def _read_data_from_file(file_data, item_start, item_end):
    array_type = file_data.data_type
    type_size = array_type.itemsize
    file_data.seek(int(item_start * type_size))
    data = file_data.read(int(type_size * (item_end - item_start)))
    array = np.frombuffer(data, dtype=array_type)
    return array


class FlashCosSearch:
    def __init__(self, max_ms2_tolerance_in_da=0.024, mz_index_step=0.0001, low_memory=False, path_data=None,
                 sqrt_transform=False):
        self.precursor_mz_array = np.zeros(0, dtype=np.float32)
        self.low_memory = low_memory
        self.sqrt_transform = sqrt_transform
        if low_memory:
            self.cos_search = FlashCosSearchCoreLowMemory(path_data=path_data,
                                                          max_ms2_tolerance_in_da=max_ms2_tolerance_in_da,
                                                          mz_index_step=mz_index_step,
                                                          sqrt_transform=sqrt_transform)
        else:
            self.cos_search = FlashCosSearchCore(path_data=path_data,
                                                 max_ms2_tolerance_in_da=max_ms2_tolerance_in_da,
                                                 mz_index_step=mz_index_step,
                                                 sqrt_transform=sqrt_transform)

    def identity_search(self, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da, **kwargs):
        """
        Run the identity search, the query spectrum should be preprocessed by `clean_spectrum()` function before calling this function.

        For super large spectral library, directly identity search is not recommended. To do the identity search on super large spectral library,
        divide the spectral library into several parts, build the index for each part, and then do the identity search on each part will be much faster.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms1_tolerance_in_da:  The MS1 tolerance in Da.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.
        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        precursor_mz_min = precursor_mz - ms1_tolerance_in_da
        precursor_mz_max = precursor_mz + ms1_tolerance_in_da
        spectra_idx_min = np.searchsorted(self.precursor_mz_array, precursor_mz_min, side='left')
        spectra_idx_max = np.searchsorted(self.precursor_mz_array, precursor_mz_max, side='right')
        if spectra_idx_min >= spectra_idx_max:
            return np.zeros(self.cos_search.total_spectra_num, dtype=np.float32), np.zeros(
                self.cos_search.total_spectra_num, dtype=np.uint32), np.zeros(self.cos_search.total_spectra_num,
                                                                              dtype=np.float32)
        else:
            return self.cos_search.search(method="open", peaks=peaks,
                                          ms2_tolerance_in_da=ms2_tolerance_in_da,
                                          search_type=1, search_spectra_idx_min=spectra_idx_min,
                                          search_spectra_idx_max=spectra_idx_max)

    def open_search(self, peaks, ms2_tolerance_in_da, **kwargs):
        """
        Run the open search, the query spectrum should be preprocessed by `clean_spectrum()` function before calling this function.

        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.
        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.cos_search.search(method="open", peaks=peaks,
                                      ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0)

    def neutral_loss_search(self, precursor_mz, peaks, ms2_tolerance_in_da, **kwargs):
        """
        Run the neutral loss search, the query spectrum should be preprocessed by `clean_spectrum()` function before calling this function.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.
        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.cos_search.search(method="neutral_loss", precursor_mz=precursor_mz, peaks=peaks,
                                      ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0)

    def hybrid_search(self, precursor_mz, peaks, ms2_tolerance_in_da, **kwargs):
        """
        Run the hybrid search, the query spectrum should be preprocessed by `clean_spectrum()` function before calling this function.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.

        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.cos_search.search_hybrid(precursor_mz=precursor_mz, peaks=peaks,
                                             ms2_tolerance_in_da=ms2_tolerance_in_da)

    def clean_spectrum_for_search(self,
                                  precursor_mz,
                                  peaks,
                                  precursor_ions_removal_da: float = 1.6,
                                  noise_threshold=0.01,
                                  min_ms2_difference_in_da: float = 0.05,
                                  max_peak_num: int = 0,
                                  sqrt_transform: bool = False):
        """
        Clean the MS/MS spectrum, need to be called before any search.

        :param precursor_mz:    The precursor m/z of the spectrum.
        :param peaks:           The peaks of the spectrum, should be a list or numpy array with shape (N, 2), N is the number of peaks. The format of the peaks is [[mz1, intensity1], [mz2, intensity2], ...].
        :param precursor_ions_removal_da:   The ions with m/z larger than precursor_mz - precursor_ions_removal_da will be removed.
                                            Default is 1.6.
        :param noise_threshold: The intensity threshold for removing the noise peaks. The peaks with intensity smaller than noise_threshold * max(intensity)
                                will be removed. Default is 0.01.
        :param min_ms2_difference_in_da:    The minimum difference between two peaks in the MS/MS spectrum. Default is 0.05.
        :param max_peak_num:    The maximum number of peaks in the MS/MS spectrum. Default is 0, which means no limit.
        :param sqrt_transform:  Whether to perform the square root transform on the intensity. Default is False.
        """
        if precursor_ions_removal_da is not None:
            max_mz = precursor_mz - precursor_ions_removal_da
        else:
            max_mz = None
        return clean_spectrum(peaks=peaks,
                              min_mz=-1,
                              max_mz=max_mz,
                              noise_threshold=noise_threshold,
                              min_ms2_difference_in_da=min_ms2_difference_in_da,
                              max_peak_num=max_peak_num,
                              normalize_intensity=True,
                              sqrt_transform=sqrt_transform)

    def search(self,
               precursor_mz, peaks,
               ms1_tolerance_in_da=0.01,
               ms2_tolerance_in_da=0.02,
               method: Union[str, List] = "all",  # identity, open, neutral_loss, hybrid, all
               precursor_ions_removal_da: Union[float, None] = 1.6,
               noise_threshold=0.01,
               min_ms2_difference_in_da: float = 0.05,
               max_peak_num: int = None):
        """
        Run the Flash search for the query spectrum.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be a list or numpy array with shape (N, 2), N is the number of peaks. The format of the peaks is [[mz1, intensity1], [mz2, intensity2], ...].
        :param ms1_tolerance_in_da:  The MS1 tolerance in Da. Default is 0.01.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da. Default is 0.02.
        :param method:  The search method, can be "identity", "open", "neutral_loss", "hybrid", "all", or list of the above.
        :param precursor_ions_removal_da:   The ions with m/z larger than precursor_mz - precursor_ions_removal_da will be removed.
                                            Default is 1.6.
        :param noise_threshold: The intensity threshold for removing the noise peaks. The peaks with intensity smaller than noise_threshold * max(intensity)
                                will be removed. Default is 0.01.
        :param min_ms2_difference_in_da:    The minimum difference between two peaks in the MS/MS spectrum. Default is 0.05.
        :param max_peak_num:    The maximum number of peaks in the MS/MS spectrum. Default is None, which means no limit.
        :return:    A dictionary with the search results. The keys are "identity_search", "open_search", "neutral_loss_search", "hybrid_search", and the values are the search results for each method.
        """
        if precursor_ions_removal_da is not None:
            max_mz = precursor_mz - precursor_ions_removal_da
        else:
            max_mz = -1

        if method == "all":
            method = {"identity", "open", "neutral_loss", "hybrid"}
        elif isinstance(method, str):
            method = {method}

        # do not normalize the intensity here, as the intensity will be normalized after alignment
        peaks = clean_spectrum(peaks=peaks,
                               min_mz=0,
                               max_mz=max_mz,
                               noise_threshold=noise_threshold,
                               min_ms2_difference_in_da=min_ms2_difference_in_da,
                               max_peak_num=max_peak_num,
                               normalize_intensity=False,
                               sqrt_transform=False)

        result = {}
        if "identity" in method:
            temp_result = self.identity_search(precursor_mz=precursor_mz,
                                               peaks=peaks,
                                               ms1_tolerance_in_da=ms1_tolerance_in_da,
                                               ms2_tolerance_in_da=ms2_tolerance_in_da)
            result["identity_search"] = _clean_search_result(temp_result)
        if "open" in method:
            temp_result = self.open_search(peaks=peaks,
                                           ms2_tolerance_in_da=ms2_tolerance_in_da)
            result["open_search"] = _clean_search_result(temp_result)
        if "neutral_loss" in method:
            temp_result = self.neutral_loss_search(precursor_mz=precursor_mz,
                                                   peaks=peaks,
                                                   ms2_tolerance_in_da=ms2_tolerance_in_da)
            result["neutral_loss_search"] = _clean_search_result(temp_result)
        if "hybrid" in method:
            temp_result = self.hybrid_search(precursor_mz=precursor_mz,
                                             peaks=peaks,
                                             ms2_tolerance_in_da=ms2_tolerance_in_da)
            result["hybrid_search"] = _clean_search_result(temp_result)
        return result

    def search_mass(self,
                    mass,
                    ms2_tolerance_in_da=0.02,
                    method: str = "open"):
        """
        Search a specific fragment or neutral loss within all library spectra. Return the matched peak intensity in the library spectra.

        :param mass: fragment mass or neutral loss mass
        :param ms2_tolerance_in_da: MS2 tolerance in Da
        :param method: open or neutral_loss
        :return:  A numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.cos_search.search_mass(query_mass=mass, ms2_tolerance_in_da=ms2_tolerance_in_da, method=method)

    def search_ref_premz_in_qry(self,
                                precursor_mz,
                                peaks,
                                mass_shift: float = 0.0,
                                ms2_tolerance_in_da=0.02,
                                method: str = "open"):
        """
        Search reference precursor m/z in query spectra. Return the matched peak intensity in the query spectra.

        :param precursor_mz: precursor m/z of the query spectra
        :param peaks: peaks of the query spectra (preprocessed)
        :param mass_shift: reference precursor m/z shift; search for the reference precursor m/z + mass_shift in query spectra
        :param ms2_tolerance_in_da: MS2 tolerance in Da
        :param method: open or neutral_loss
        :return:  A numpy array with shape (N,), N is the number of spectra in the library.
        """
        # arr to return
        ref_intensity_arr = np.zeros(len(self.precursor_mz_array), dtype=np.float32)

        if len(peaks) > 0:
            # shift the precursor m/z
            target_mass_arr = self.precursor_mz_array + mass_shift

            peaks[:, 1] = peaks[:, 1] / max(peaks[:, 1])

            if method == "open":
                qry_mass_arr = peaks[:, 0]
            else:
                qry_mass_arr = precursor_mz - peaks[:, 0]

            # Compute the absolute difference matrix
            diff_matrix = np.abs(qry_mass_arr[:, np.newaxis] - target_mass_arr)

            # Find the minimum difference for each target mass
            min_diffs = np.min(diff_matrix, axis=0)

            # Find the index of the minimum difference for each target mass
            min_diff_idx = np.argmin(diff_matrix, axis=0)

            # Mask where the minimum difference exceeds the tolerance
            valid_matches = min_diffs <= ms2_tolerance_in_da
            matched_indices = min_diff_idx[valid_matches]

            # Assign intensities for valid matches
            ref_intensity_arr[valid_matches] = peaks[matched_indices, 1]

        return ref_intensity_arr

    def build_index(self,
                    all_spectra_list: list = None,
                    max_indexed_mz: float = 1500.00005,
                    precursor_ions_removal_da: Union[float, None] = 1.6,
                    noise_threshold=0.01,
                    min_ms2_difference_in_da: float = 0.05,
                    max_peak_num: int = 0,
                    clean_spectra: bool = True):
        """
        Set the library spectra for search.

        The `all_spectra_list` must be a list of dictionaries, with each dictionary containing at least two keys: "precursor_mz" and "peaks".
        The dictionary should be in the format of {"precursor_mz": precursor_mz, "peaks": peaks, ...}, All keys in the dictionary, except "peaks,"
        will be saved as the metadata and can be accessed using the  __getitem__ function (e.g. cosine_search[0] returns the metadata for the
        first spectrum in the library).

            - The precursor_mz is the precursor m/z value of the MS/MS spectrum;

            - The peaks is a numpy array or a nested list of the MS/MS spectrum, looks like [[mz1, intensity1], [mz2, intensity2], ...].

        :param all_spectra_list:    A list of dictionaries in the format of {"precursor_mz": precursor_mz, "peaks": peaks},
                                    the spectra in the list do not need to be sorted by the precursor m/z.
                                    This function will sort the spectra by the precursor m/z and output the sorted spectra list.

        :param max_indexed_mz: The maximum m/z value that will be indexed. Default is 1500.00005.
        :param precursor_ions_removal_da:   The ions with m/z larger than precursor_mz - precursor_ions_removal_da will be removed.
                                            Default is 1.6.
        :param noise_threshold: The intensity threshold for removing the noise peaks. The peaks with intensity smaller than noise_threshold * max(intensity)
                                will be removed. Default is 0.01.
        :param min_ms2_difference_in_da:    The minimum difference between two peaks in the MS/MS spectrum. Default is 0.05.
        :param max_peak_num:    The maximum number of peaks in the MS/MS spectrum. Default is 0, which means no limit.
        :param clean_spectra:   If True, the spectra will be cleaned before indexing. Default is True. If ALL spectra in the library are pre-cleaned with the
                                function `clean_spectrum` or `clean_spectrum_for_search`, set this parameter to False. ALWAYS set this parameter to true if
                                the spectra are not pre-prepossessed with the function `clean_spectrum` or `clean_spectrum_for_search`.
        :return:    If the all_spectra_list is provided, this function will return the sorted spectra list.
        """

        # Sort the spectra by the precursor m/z.
        all_sorted_spectra_list = sorted(all_spectra_list, key=lambda x: x["precursor_mz"])

        # Clean the spectra, and collect the non-empty spectra
        all_spectra_list = []
        all_metadata_list = []
        for spec in all_sorted_spectra_list:
            # Clean the peaks
            if clean_spectra:
                spec["peaks"] = self.clean_spectrum_for_search(peaks=spec["peaks"],
                                                               precursor_mz=spec["precursor_mz"],
                                                               precursor_ions_removal_da=precursor_ions_removal_da,
                                                               noise_threshold=noise_threshold,
                                                               min_ms2_difference_in_da=min_ms2_difference_in_da,
                                                               max_peak_num=max_peak_num,
                                                               sqrt_transform=self.sqrt_transform)
            if len(spec["peaks"]) > 0:
                all_spectra_list.append(spec)
                all_metadata_list.append(pickle.dumps(spec))

        # Extract precursor m/z array
        self.precursor_mz_array = np.array([spec["precursor_mz"] for spec in all_spectra_list], dtype=np.float32)

        # Extract metadata array
        all_metadata_len = np.array([0] + [len(metadata) for metadata in all_metadata_list], dtype=np.uint64)
        self.metadata_loc = np.cumsum(all_metadata_len).astype(np.uint64)
        self.metadata = np.frombuffer(b''.join(all_metadata_list), dtype=np.uint8)

        # Call father class to build the index.
        self.cos_search.build_index(all_spectra_list, max_indexed_mz)
        return all_spectra_list

    def __getitem__(self, index):
        """
        Get the MS/MS metadata by the index.

        :param index:   The index of the MS/MS spectrum.
        :return:    The MS/MS spectrum in the format of {"precursor_mz": precursor_mz, "peaks": peaks}.
        """
        if self.metadata is not None:
            spectrum = pickle.loads(self.metadata[self.metadata_loc[index]:self.metadata_loc[index + 1]].tobytes())
        else:
            spectrum = {"precursor_mz": self.precursor_mz_array[index]}
        return spectrum

    def write(self, path_data=None):
        """
        Write the MS/MS spectral library to a file.

        :param path_data:   The path of the file to write.
        :return:    None
        """
        if path_data is None:
            path_data = self.cos_search.path_data
        else:
            path_data = Path(path_data)

        path_data = Path(path_data)
        path_data.mkdir(parents=True, exist_ok=True)

        self.precursor_mz_array.tofile(str(path_data / "precursor_mz.npy"))
        self.metadata.tofile(str(path_data / "metadata.npy"))
        self.metadata_loc.tofile(str(path_data / "metadata_loc.npy"))

        self.cos_search.write(path_data)

    def read(self, path_data=None):
        """
        Read the MS/MS spectral library from a file.

        :param path_data:   The path of the file to read.
        :return:    None
        """
        if path_data is None:
            path_data = self.cos_search.path_data
        else:
            path_data = Path(path_data)

        if self.low_memory:
            self.precursor_mz_array = np.memmap(path_data / "precursor_mz.npy", dtype=np.float32, mode="r")
            self.metadata = np.memmap(path_data / "metadata.npy", dtype=np.uint8, mode="r")
            self.metadata_loc = np.memmap(path_data / "metadata_loc.npy", dtype=np.uint64, mode="r")
        else:
            self.precursor_mz_array = np.fromfile(str(path_data / "precursor_mz.npy"), dtype=np.float32)
            self.metadata = np.fromfile(str(path_data / "metadata.npy"), dtype=np.uint8)
            self.metadata_loc = np.fromfile(str(path_data / "metadata_loc.npy"), dtype=np.uint64)

        return self.cos_search.read(path_data)

    def save_memory_for_multiprocessing(self):
        """
        Save the memory for multiprocessing. This function will move the numpy array in the index to shared memory in order to save memory.

        This function is not required when you only use one thread to search the MS/MS spectra.
        When use multiple threads, this function is also not required but highly recommended, as it avoids the memory copy and saves a lot of memory and time.

        :return:    None
        """
        self.cos_search.save_memory_for_multiprocessing()


def _clean_search_result(temp_result):
    score_arr = temp_result[0]
    score_arr = np.where(np.isfinite(score_arr), score_arr, 0)
    temp_result = (score_arr, temp_result[1], temp_result[2])
    return temp_result


def clean_spectrum(
        peaks,
        min_mz: float = -1.,
        max_mz: float = -1.,
        noise_threshold: float = 0.0,
        min_ms2_difference_in_da: float = 0.05,
        min_ms2_difference_in_ppm: float = -1.,
        max_peak_num: int = -1,
        normalize_intensity: bool = True,
        sqrt_transform: bool = True,
        **kwargs
) -> np.ndarray:
    """
    Clean, centroid, and normalize a spectrum with the following steps:

        1. Remove empty peaks (m/z <= 0 or intensity <= 0).
        2. Remove peaks with m/z >= max_mz or m/z < min_mz.
        3. Centroid the spectrum by merging peaks within min_ms2_difference_in_da.
        4. Remove peaks with intensity < noise_threshold * max_intensity.
        5. Keep only the top max_peak_num peaks.
        6. Normalize the intensity. sum of squared intensity = 1.


    Parameters
    ----------
    peaks : np.ndarray in shape (n_peaks, 2), dtype=np.float32 or list[list[float, float]]
        A 2D array of shape (n_peaks, 2) where the first column is m/z and the second column is intensity.

    min_mz : float, optional
        The minimum m/z to keep. Defaults to None, which will skip removing peaks with m/z < min_mz.

    max_mz : float, optional
        The maximum m/z to keep. Defaults to None, which will skip removing peaks with m/z >= max_mz.

    noise_threshold : float, optional
        The minimum intensity to keep. Defaults to 0.01, which will remove peaks with intensity < 0.01 * max_intensity.

    min_ms2_difference_in_da : float, optional
        The minimum m/z difference between two peaks in the resulting spectrum. Defaults to 0.05, which will merge peaks within 0.05 Da. If a negative value is given, the min_ms2_difference_in_ppm will be used instead.

    min_ms2_difference_in_ppm : float, optional
        The minimum m/z difference between two peaks in the resulting spectrum. Defaults to -1, which will use the min_ms2_difference_in_da instead. If a negative value is given, the min_ms2_difference_in_da will be used instead.
        ** Note either min_ms2_difference_in_da or min_ms2_difference_in_ppm must be positive. If both are positive, min_ms2_difference_in_ppm will be used. **

    max_peak_num : int, optional
        The maximum number of peaks to keep. Defaults to None, which will keep all peaks.

    normalize_intensity : bool, optional
        Whether to normalize the intensity. Defaults to True. If False, the intensity will be kept as is.

    sqrt_transform : bool, optional
        Whether to square root the intensity. Defaults to True. If False, the intensity will be kept as is.

    **kwargs : optional
        Those keyword arguments will be ignored.

        _

    Returns
    -------
    np.ndarray in shape (n_peaks, 2), dtype=np.float32
        The cleaned spectrum will be guaranteed to be sorted by m/z in ascending order.

    """
    # Check the input spectrum and convert it to numpy array with shape (n, 2) and dtype np.float32.
    peaks = np.asarray(peaks, dtype=np.float32, order="C")
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    assert peaks.ndim == 2 and peaks.shape[1] == 2, "The input spectrum must be a 2D numpy array with shape (n, 2)."

    # Step 1. Remove empty peaks (m/z <= 0 or intensity <= 0).
    peaks = peaks[np.bitwise_and(peaks[:, 0] > 0, peaks[:, 1] > 0)]

    # Step 2. Remove peaks with m/z >= max_mz or m/z < min_mz.
    if min_mz is not None and min_mz > 0:
        peaks = peaks[peaks[:, 0] >= min_mz]

    if max_mz is not None and max_mz > 0:
        peaks = peaks[peaks[:, 0] <= max_mz]

    if peaks.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Step 3. Centroid the spectrum by merging peaks within min_ms2_difference_in_da.
    # The peaks will be sorted by m/z in ascending order.
    if min_ms2_difference_in_ppm > 0:
        peaks = centroid_spectrum(peaks, ms2_da=-1, ms2_ppm=min_ms2_difference_in_ppm)
    elif min_ms2_difference_in_da > 0:
        peaks = centroid_spectrum(peaks, ms2_da=min_ms2_difference_in_da, ms2_ppm=-1)
    else:
        raise ValueError("Either min_ms2_difference_in_da or min_ms2_difference_in_ppm must be positive.")

    # Step 4. Remove peaks with intensity < noise_threshold * max_intensity.
    if noise_threshold is not None:
        peaks = peaks[peaks[:, 1] >= noise_threshold * np.max(peaks[:, 1])]

    # Step 5. Keep only the top max_peak_num peaks.
    if max_peak_num is not None and max_peak_num > 0:
        # Sort the spectrum by intensity.
        peaks = peaks[np.argsort(peaks[:, 1])[-max_peak_num:]]
        # Sort the spectrum by m/z.
        peaks = peaks[np.argsort(peaks[:, 0])]

    # Step 6. Normalize the intensity. sum of squared intensity = 1.
    if normalize_intensity:
        if sqrt_transform:
            # square root the intensity to reduce the effect of large peaks.
            peaks[:, 1] = np.sqrt(peaks[:, 1])
        # normalize
        spectrum_sum = np.sqrt(np.sum(np.square(peaks[:, 1])))
        if spectrum_sum > 0:
            peaks[:, 1] = peaks[:, 1] / spectrum_sum
        else:
            peaks = np.zeros((0, 2), dtype=np.float32)

    return peaks


def centroid_spectrum(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> np.ndarray:
    """
    Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da.

    Parameters
    ----------
    peaks : np.ndarray in shape (n_peaks, 2), dtype=np.float32
        A 2D array of shape (n_peaks, 2) where the first column is m/z and the second column is intensity.

    ms2_ppm : float, optional
        The m/z tolerance in ppm. Defaults to -1, which will use ms2_da instead.

    ms2_da : float, optional
        The m/z tolerance in Da. Defaults to -1, which will use ms2_ppm instead.

        **Note: ms2_ppm and ms2_da cannot be both negative, if so, an Exception will be raised. If both are positive, ms2_da will be used.**

    Returns
    -------
    np.ndarray in shape (n_peaks, 2), dtype=np.float32
        The resulting spectrum after centroiding will be guaranteed to have been sorted by m/z, and the difference between two peaks will be >= ms2_ppm or >= ms2_da.
    """

    # Sort the peaks by m/z.
    peaks = peaks[np.argsort(peaks[:, 0])]
    is_centroided: int = _check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    while is_centroided == 0:
        peaks = _centroid_spectrum(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
        is_centroided = _check_centroid(peaks, ms2_da=ms2_da, ms2_ppm=ms2_ppm)
    return peaks


def _centroid_spectrum(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> np.ndarray:
    """Centroid a spectrum by merging peaks within the +/- ms2_ppm or +/- ms2_da."""
    # Construct a new spectrum to avoid memory reallocation.
    peaks_new = np.zeros((peaks.shape[0], 2), dtype=np.float32)
    peaks_new_len = 0

    # Get the intensity argsort order.
    intensity_order = np.argsort(peaks[:, 1])

    mz_delta_allowed_left = ms2_da
    mz_delta_allowed_right = ms2_da
    # Iterate through the peaks from high to low intensity.
    for idx in intensity_order[::-1]:
        if ms2_ppm > 0:
            mz_delta_allowed_left = peaks[idx, 0] * ms2_ppm * 1e-6
            # For the right boundary, the mz_delta_allowed_right = peaks[right_idx, 0] * ms2_ppm * 1e-6 = peaks[idx, 0] / (1 - ms2_ppm * 1e-6)
            mz_delta_allowed_right = peaks[idx, 0] / (1 - ms2_ppm * 1e-6)

        if peaks[idx, 1] > 0:
            # Find the left boundary.
            left_idx = idx - 1
            while left_idx >= 0 and peaks[idx, 0] - peaks[left_idx, 0] <= mz_delta_allowed_left:
                left_idx -= 1
            left_idx += 1

            # Find the right boundary.
            right_idx = idx + 1
            while right_idx < peaks.shape[0] and peaks[right_idx, 0] - peaks[idx, 0] <= mz_delta_allowed_right:
                right_idx += 1

            # Merge the peaks within the boundary. The new m/z is the weighted average of the m/z and intensity.
            intensity_sum = np.sum(peaks[left_idx:right_idx, 1])
            intensity_weighted_mz_sum = np.sum(peaks[left_idx:right_idx, 1] * peaks[left_idx:right_idx, 0])
            peaks_new[peaks_new_len, 0] = intensity_weighted_mz_sum / intensity_sum
            peaks_new[peaks_new_len, 1] = intensity_sum
            peaks_new_len += 1

            # Set the intensity of the merged peaks to 0.
            peaks[left_idx:right_idx, 1] = 0

    # Return the new spectrum.
    peaks_new = peaks_new[:peaks_new_len]
    return peaks_new[np.argsort(peaks_new[:, 0])]


def _check_centroid(peaks: np.ndarray, ms2_da: float = -1, ms2_ppm: float = -1) -> int:
    """Check if the spectrum is centroided. 0 for False and 1 for True."""
    if peaks.shape[0] <= 1:
        return 1

    if ms2_ppm >= 0:
        # Use ms2_ppm to check if the spectrum is centroided.
        return 1 if np.all(np.diff(peaks[:, 0]) >= peaks[1:, 0] * ms2_ppm * 1e-6) else 0
    elif ms2_da >= 0:
        # Use ms2_da to check if the spectrum is centroided.
        return 1 if np.all(np.diff(peaks[:, 0]) >= ms2_da) else 0


# test
if __name__ == "__main__":
    # cosine between two vectors
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


    print(cosine_similarity(np.array([1, 1, 1, 0, 0]), np.array([1, 2, 4, 3, 10])))

    print(cosine_similarity(np.sqrt(np.array([1, 1, 1, 0, 0])), np.sqrt(np.array([1, 2, 4, 3, 10]))))

    # load spectral library
    spectral_library = [{
        "id": "Demo spectrum 1",
        "precursor_mz": 150.0,
        "peaks": [[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]]
    }, {
        "id": "Demo spectrum 2",
        "precursor_mz": 200.0,
        "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32),
        "metadata": "ABC"
    }, {
        "id": "Demo spectrum 3",
        "precursor_mz": 250.0,
        "peaks": np.array([[200.0, 1.0], [201.0, 1.0], [202.0, 1.0]], dtype=np.float32),
        "XXX": "YYY",
    }, {
        "precursor_mz": 350.0,
        "peaks": [[100.0, 1.0], [101.0, 1.0], [302.0, 1.0]]}]

    ms2_tol = 0.05
    # cos search
    cos_search = FlashCosSearch(max_ms2_tolerance_in_da=ms2_tol * 1.05,
                                mz_index_step=0.0001,
                                low_memory=False,
                                sqrt_transform=True)
    cos_search.build_index(spectral_library,
                           max_indexed_mz=1500,
                           precursor_ions_removal_da=None,
                           noise_threshold=0.0,
                           min_ms2_difference_in_da=ms2_tol * 2.2,
                           max_peak_num=-1,
                           clean_spectra=True)

    cos_result = cos_search.search(
        precursor_mz=400.0,
        peaks=[[200.0, 1.0], [201.0, 2.0], [202.0, 4.0], [203.0, 3.0], [303.0, 10.0]],
        ms1_tolerance_in_da=0.02,
        ms2_tolerance_in_da=ms2_tol,
        method="open",  # "identity", "open", "neutral_loss", "hybrid", "all", or list of the above
        precursor_ions_removal_da=None,
        noise_threshold=0.0,
        min_ms2_difference_in_da=ms2_tol * 2.2,
        max_peak_num=None
    )

    score_arr, matched_peak_arr, spec_usage_arr = cos_result['open_search']

    print(cos_result)

    # # Get the metadata of library spectrum
    # # NOTE: library spectra are resorted during 'build index', need to get the metadata by the index.
    # for i in range(len(spectral_library)):
    #     print(cos_search[i])
