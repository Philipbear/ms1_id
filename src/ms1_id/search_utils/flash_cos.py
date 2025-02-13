"""
Flash search for cos / reverse cos
return matched peaks, spectral usage, reverse score


This file contains code modified from the FlashEntropySearch project
(https://github.com/YuanyueLi/FlashEntropySearch)
Copyright Yuanyue Li in 2023

Modified by Shipei Xing in 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import multiprocessing
import pickle
from functools import reduce
from pathlib import Path
from typing import Union, List

import numpy as np

from ms1_id.search_utils.preprocess_ms2 import preprocess_ms2

np.seterr(divide='ignore', invalid='ignore')


class FlashCosCore:
    def __init__(self, path_data=None, max_ms2_tolerance_in_da=0.024, mz_index_step=0.0001,
                 peak_scale_k=8.0,
                 peak_intensity_power=1.0):
        """
        Initialize the search core. for CPU only.
        :param path_array: The path array of the index files.
        :param max_ms2_tolerance_in_da: The maximum MS2 tolerance used when searching the MS/MS spectra, in Dalton. Default is 0.024.
        :param mz_index_step:   The step size of the m/z index, in Dalton. Default is 0.0001.
                                The smaller the step size, the faster the search, but the larger the index size and longer the index building time.
        """
        self.mz_index_step = mz_index_step
        self._init_for_multiprocessing = False
        self.max_ms2_tolerance_in_da = max_ms2_tolerance_in_da
        self.peak_scale_k = peak_scale_k
        self.peak_intensity_power = peak_intensity_power

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
            search_spectra_idx_max=0,
            reverse=False
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
        # Calculate the similarity for this matched peak
        similarity_arr = np.zeros(self.total_spectra_num, dtype=np.float32)
        matched_cnt_arr = np.zeros(self.total_spectra_num, dtype=np.uint32)
        spec_usage_arr = np.zeros(self.total_spectra_num, dtype=np.float32)  # sum(matched peaks intensity) / sum(query peaks intensity)

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
            spec_usage_arr = np.sum(match_table_q, axis=0) / np.sum(peaks[:, 1])

            # transform query spectrum intensity in each column
            match_table_q = np.power(match_table_q, self.peak_intensity_power)
            peaks[:, 1] = np.power(peaks[:, 1], self.peak_intensity_power)

            if reverse:
                match_table_q = match_table_q / np.sqrt(np.sum(np.square(match_table_q), axis=0))
                # calculate similarity
                similarity_arr = np.sum(match_table_q * match_table_r, axis=0)

            else:
                match_table_q = match_table_q / np.sqrt(np.sum(np.square(peaks[:, 1]), axis=0))
                # calculate similarity
                similarity_arr = np.sum(match_table_q * match_table_r, axis=0)

            # count matched peaks
            matched_cnt_arr = np.sum(match_table_q > 0, axis=0)
        elif search_type == 1:
            # search type 1: search a range of the library, for identity search
            match_table_q = match_table_q[:, search_spectra_idx_min:search_spectra_idx_max]
            match_table_r = match_table_r[:, search_spectra_idx_min:search_spectra_idx_max]
            # calculate spectral usage
            part_spectral_usage = np.sum(match_table_q, axis=0) / np.sum(peaks[:, 1])

            # normalize query spectrum intensity in each column
            match_table_q = np.power(match_table_q, self.peak_intensity_power)
            peaks[:, 1] = np.power(peaks[:, 1], self.peak_intensity_power)

            if reverse:
                match_table_q_rev = match_table_q / np.sqrt(np.sum(np.square(match_table_q), axis=0))
                # calculate similarity
                part_similarity = np.sum(match_table_q_rev * match_table_r, axis=0)

            else:
                match_table_q = match_table_q / np.sqrt(np.sum(np.square(peaks[:, 1])))
                # calculate similarity
                part_similarity = np.sum(match_table_q * match_table_r, axis=0)

            # count matched peaks
            part_matched_cnt = np.sum(match_table_q > 0, axis=0)

            # update the similarity and matched count
            similarity_arr[search_spectra_idx_min:search_spectra_idx_max] = part_similarity
            matched_cnt_arr[search_spectra_idx_min:search_spectra_idx_max] = part_matched_cnt
            spec_usage_arr[search_spectra_idx_min:search_spectra_idx_max] = part_spectral_usage

        del match_table_q, match_table_r

        return similarity_arr, matched_cnt_arr, spec_usage_arr

    def search_hybrid(self, precursor_mz=None, peaks=None, ms2_tolerance_in_da=0.02, reverse=False):
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
                    np.zeros(self.total_spectra_num, dtype=np.uint32),
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

        # Go through all the peaks in the spectrum and calculate the similarity
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

            # # Check if this query peak is already matched to a product ion in the same library spectrum
            # duplicate_idx_in_nl = self._remove_duplicate_with_cpu(modified_idx, modified_idx_nl,
            #                                                       self.total_spectra_num)
            # intensity_library_nl[duplicate_idx_in_nl] = 0

            # fill in the match table for neutral loss
            match_table_q_nl[k, modified_idx_nl] = intensity
            match_table_r_nl[k, modified_idx_nl] = intensity_library_nl

        # merge the match table
        match_table_q_all = np.maximum(match_table_q, match_table_q_nl)
        match_table_r_all = np.maximum(match_table_r, match_table_r_nl)

        del match_table_q, match_table_r, match_table_q_nl, match_table_r_nl

        # calculate spectral usage
        spec_usage_arr = np.sum(match_table_q_all, axis=0) / np.sum(peaks[:, 1])

        # normalize query spectrum intensity in each column
        match_table_q_all = np.power(match_table_q_all, self.peak_intensity_power)
        peaks[:, 1] = np.power(peaks[:, 1], self.peak_intensity_power)

        if reverse:
            match_table_q_all_rev = match_table_q_all / np.sqrt(np.sum(np.square(match_table_q_all), axis=0))
            # calculate similarity
            similarity_arr = np.sum(match_table_q_all_rev * match_table_r_all, axis=0)

        else:
            match_table_q_all = match_table_q_all / np.sqrt(np.sum(np.square(peaks[:, 1]), axis=0))
            # calculate similarity
            similarity_arr = np.sum(match_table_q_all * match_table_r_all, axis=0)

        # count matched peaks
        matched_cnt_arr = np.sum(match_table_q_all > 0, axis=0)

        return similarity_arr, matched_cnt_arr, spec_usage_arr

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

        print('Number of spectra:', len(all_spectra_list))

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


def _read_data_from_file(file_data, item_start, item_end):
    array_type = file_data.data_type
    type_size = array_type.itemsize
    file_data.seek(int(item_start * type_size))
    data = file_data.read(int(type_size * (item_end - item_start)))
    array = np.frombuffer(data, dtype=array_type)
    return array


class FlashCos:
    def __init__(self, max_ms2_tolerance_in_da=0.024, mz_index_step=0.0001, path_data=None,
                 peak_scale_k=8.0,
                 peak_intensity_power=1.0):
        self.precursor_mz_array = np.zeros(0, dtype=np.float32)
        self.peak_scale_k = peak_scale_k
        self.peak_intensity_power = peak_intensity_power

        self.similarity_search = FlashCosCore(path_data=path_data,
                                              max_ms2_tolerance_in_da=max_ms2_tolerance_in_da,
                                              mz_index_step=mz_index_step,
                                              peak_scale_k=peak_scale_k,
                                              peak_intensity_power=peak_intensity_power)

    def identity_search(self, precursor_mz, peaks, ms1_tolerance_in_da, ms2_tolerance_in_da,
                        reverse, **kwargs):
        """
        Run the identity search, the query spectrum should be preprocessed.

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
            return (np.zeros(self.similarity_search.total_spectra_num, dtype=np.float32),
                    np.zeros(self.similarity_search.total_spectra_num, dtype=np.float32),
                    np.zeros(self.similarity_search.total_spectra_num, dtype=np.float32),
                    np.zeros(self.similarity_search.total_spectra_num, dtype=np.float32))
        else:
            return self.similarity_search.search(method="open", peaks=peaks, precursor_mz=precursor_mz,
                                                 ms2_tolerance_in_da=ms2_tolerance_in_da,
                                                 search_type=1, search_spectra_idx_min=spectra_idx_min,
                                                 search_spectra_idx_max=spectra_idx_max,
                                                 reverse=reverse)

    def open_search(self, precursor_mz, peaks, ms2_tolerance_in_da, reverse, **kwargs):
        """
        Run the open search, the query spectrum should be preprocessed.

        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.
        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.similarity_search.search(method="open", peaks=peaks, precursor_mz=precursor_mz,
                                             ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0,
                                             reverse=reverse)

    def neutral_loss_search(self, precursor_mz, peaks, ms2_tolerance_in_da, reverse, **kwargs):
        """
        Run the neutral loss search, the query spectrum should be preprocessed.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.
        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.similarity_search.search(method="neutral_loss", precursor_mz=precursor_mz, peaks=peaks,
                                             ms2_tolerance_in_da=ms2_tolerance_in_da, search_type=0,
                                             reverse=reverse)

    def hybrid_search(self, precursor_mz, peaks, ms2_tolerance_in_da, reverse, **kwargs):
        """
        Run the hybrid search, the query spectrum should be preprocessed.

        :param precursor_mz:    The precursor m/z of the query spectrum.
        :param peaks:           The peaks of the query spectrum, should be the output of `clean_spectrum()` function.
        :param ms2_tolerance_in_da:  The MS2 tolerance in Da.

        :return:    The similarity score for each spectrum in the library, a numpy array with shape (N,), N is the number of spectra in the library.
        """
        return self.similarity_search.search_hybrid(precursor_mz=precursor_mz, peaks=peaks,
                                                    ms2_tolerance_in_da=ms2_tolerance_in_da,
                                                    reverse=reverse)

    def clean_spectrum_for_search(self,
                                  precursor_mz,
                                  peaks,
                                  precursor_ions_removal_da: float = 1.6,
                                  noise_threshold=0.01,
                                  min_ms2_difference_in_da: float = 0.05,
                                  peak_scale_k: float = 8.0,
                                  peak_intensity_power: float = 1.0):
        """
        Clean the MS/MS spectrum, need to be called before any search.

        :param precursor_mz:    The precursor m/z of the spectrum.
        :param peaks:           The peaks of the spectrum, should be a list or numpy array with shape (N, 2), N is the number of peaks. The format of the peaks is [[mz1, intensity1], [mz2, intensity2], ...].
        :param precursor_ions_removal_da:   The ions with m/z larger than precursor_mz - precursor_ions_removal_da will be removed.
                                            Default is 1.6.
        :param noise_threshold: The intensity threshold for removing the noise peaks. The peaks with intensity smaller than noise_threshold * max(intensity)
                                will be removed. Default is 0.01.
        :param min_ms2_difference_in_da:    The minimum difference between two peaks in the MS/MS spectrum. Default is 0.05.
        :param peak_intensity_power:    The power of the peak intensity. Default is 1.0.
        """

        if precursor_ions_removal_da is not None:
            max_mz = precursor_mz - precursor_ions_removal_da
        else:
            max_mz = None

        return preprocess_ms2(peaks=peaks,
                              prec_mz=precursor_mz,
                              min_mz=-1,
                              max_mz=max_mz,
                              relative_intensity_cutoff=noise_threshold,
                              min_ms2_difference_in_da=min_ms2_difference_in_da,
                              min_ms2_difference_in_ppm=-1,
                              top6_every_50da=True,
                              peak_scale_k=peak_scale_k,
                              peak_intensity_power=peak_intensity_power,
                              peak_norm='sum_sq')

    def search(self,
               precursor_mz, peaks,
               ms1_tolerance_in_da=0.01,
               ms2_tolerance_in_da=0.02,
               method: Union[str, List] = "all",  # identity, open, neutral_loss, hybrid, all
               precursor_ions_removal_da: Union[float, None] = 1.6,
               noise_threshold=0.01,
               min_ms2_difference_in_da: float = 0.05,
               reverse: bool = True):
        """
        Run the Flash search for the query spectrum.
        :return:    A dictionary with the search results. The keys are "identity_search", "open_search", "neutral_loss_search", "hybrid_search", and the values are the search results for each method.
        """

        if method == "all":
            method = {"identity", "open", "neutral_loss", "hybrid"}
        elif isinstance(method, str):
            method = {method}

        if precursor_ions_removal_da is not None:
            _max_mz = precursor_mz - precursor_ions_removal_da
        else:
            _max_mz = None

        # do not normalize the intensity here, as the intensity will be normalized after alignment
        peaks = preprocess_ms2(peaks=peaks,
                               prec_mz=precursor_mz,
                               min_mz=-1,
                               max_mz=_max_mz,
                               relative_intensity_cutoff=noise_threshold,
                               min_ms2_difference_in_da=min_ms2_difference_in_da,
                               min_ms2_difference_in_ppm=-1,
                               top6_every_50da=False,
                               peak_intensity_power=1.0,
                               peak_norm=None)

        result = {}
        if "identity" in method:
            temp_result = self.identity_search(precursor_mz=precursor_mz,
                                               peaks=peaks,
                                               ms1_tolerance_in_da=ms1_tolerance_in_da,
                                               ms2_tolerance_in_da=ms2_tolerance_in_da,
                                               reverse=reverse)
            result["identity_search"] = _clean_search_result(temp_result)
        if "open" in method:
            temp_result = self.open_search(precursor_mz=precursor_mz,
                                           peaks=peaks,
                                           ms2_tolerance_in_da=ms2_tolerance_in_da,
                                           reverse=reverse)
            result["open_search"] = _clean_search_result(temp_result)
        if "neutral_loss" in method:
            temp_result = self.neutral_loss_search(precursor_mz=precursor_mz,
                                                   peaks=peaks,
                                                   ms2_tolerance_in_da=ms2_tolerance_in_da,
                                                   reverse=reverse)
            result["neutral_loss_search"] = _clean_search_result(temp_result)
        if "hybrid" in method:
            temp_result = self.hybrid_search(precursor_mz=precursor_mz,
                                             peaks=peaks,
                                             ms2_tolerance_in_da=ms2_tolerance_in_da,
                                             reverse=reverse)
            result["hybrid_search"] = _clean_search_result(temp_result)
        return result

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
                    clean_spectra: bool = True):
        """
        Set the library spectra for search. This function will sort the spectra by the precursor m/z and output the sorted spectra list.

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
                                                               peak_scale_k=self.peak_scale_k,
                                                               peak_intensity_power=self.peak_intensity_power)

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
        self.similarity_search.build_index(all_spectra_list, max_indexed_mz)
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
        """
        if path_data is None:
            path_data = self.similarity_search.path_data
        else:
            path_data = Path(path_data)

        path_data = Path(path_data)
        path_data.mkdir(parents=True, exist_ok=True)

        self.precursor_mz_array.tofile(str(path_data / "precursor_mz.npy"))
        self.metadata.tofile(str(path_data / "metadata.npy"))
        self.metadata_loc.tofile(str(path_data / "metadata_loc.npy"))

        self.similarity_search.write(path_data)

    def read(self, path_data=None):
        """
        Read the MS/MS spectral library from a file.
        """
        if path_data is None:
            path_data = self.similarity_search.path_data
        else:
            path_data = Path(path_data)

        self.precursor_mz_array = np.fromfile(str(path_data / "precursor_mz.npy"), dtype=np.float32)
        self.metadata = np.fromfile(str(path_data / "metadata.npy"), dtype=np.uint8)
        self.metadata_loc = np.fromfile(str(path_data / "metadata_loc.npy"), dtype=np.uint64)

        return self.similarity_search.read(path_data)

    def save_memory_for_multiprocessing(self):
        """
        Save the memory for multiprocessing. This function will move the numpy array in the index to shared memory in order to save memory.

        This function is not required when you only use one thread to search the MS/MS spectra.
        When use multiple threads, this function is also not required but highly recommended, as it avoids the memory copy and saves a lot of memory and time.

        :return:    None
        """
        self.similarity_search.save_memory_for_multiprocessing()


def _clean_search_result(temp_result):
    score_arr = np.where(np.isfinite(temp_result[0]), temp_result[0], 0)
    temp_result = (score_arr, temp_result[1], temp_result[2])
    return temp_result


# test
if __name__ == "__main__":
    # cosine between two vectors
    def scale(peaks, prec_mz):
        scaling_factor = peaks[:, 0] / prec_mz * 8.0
        # print('scaling_factor:', scaling_factor)
        return peaks[:, 1] * np.exp(scaling_factor)


    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # load spectral library
    spectral_library = [{
        "id": "Demo spectrum 1",
        "precursor_mz": 150.0,
        "peaks": [[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]]
    }, {
        "id": "Demo spectrum 2",
        "precursor_mz": 200.0,
        "peaks": np.array([[100.0, 10], [101.0, 5], [102.0, 20]], dtype=np.float32),
        "metadata": "ABC"
    }, {
        "id": "Demo spectrum 3",
        "precursor_mz": 250.0,
        "peaks": np.array([[100.0, 100], [101.0, 0.5], [202.0, 20], [205.0, 20]], dtype=np.float32),
        "XXX": "YYY",
    }, {
        "precursor_mz": 350.0,
        "peaks": [[200.0, 1.0], [250.0, 2.0], [349.0, 1.0]]}]

    ms2_tol = 0.02
    # search
    search_eng = FlashCos(max_ms2_tolerance_in_da=ms2_tol * 1.05,
                          mz_index_step=0.0001,
                          peak_scale_k=8.0,
                          peak_intensity_power=0.5)
    search_eng.build_index(spectral_library,
                           max_indexed_mz=2000,
                           precursor_ions_removal_da=0.5,
                           noise_threshold=0.0,
                           min_ms2_difference_in_da=ms2_tol * 2.2,
                           clean_spectra=True)

    # test
    _precursor_mz = 250.0
    peaks = np.array([[100.0, 100], [101.0, 0.5], [202.0, 20], [203.0, 0], [204.0, 0], [205.0, 20]])

    print(cosine_similarity(np.array([100, 0.5, 20, 0, 0, 20]), np.array([100, 30, 4, 3, 10, 0])))
    # print(scale(peaks, _precursor_mz))
    print(cosine_similarity(scale(peaks, _precursor_mz), np.array([100, 30, 4, 3, 10, 0])))
    print(cosine_similarity(np.sqrt(scale(peaks, _precursor_mz)), np.sqrt(np.array([100, 30, 4, 3, 10, 0]))))

    print('reverse')
    _peaks = np.array([[100.0, 100], [101.0, 0.5], [202.0, 20], [205.0, 20]])
    # print(scale(_peaks, _precursor_mz))
    print(cosine_similarity(scale(_peaks, _precursor_mz), np.array([100, 30, 4, 0])))
    print(cosine_similarity(np.sqrt(scale(_peaks, _precursor_mz)), np.sqrt(np.array([100, 30, 4, 0]))))

    search_result = search_eng.search(
        precursor_mz=250.0,
        peaks=np.array([[100.0, 100], [101.0, 30], [202.0, 4.0], [203.0, 3.0], [204.0, 10], [205.0, 0]]),
        ms1_tolerance_in_da=0.02,
        ms2_tolerance_in_da=ms2_tol,
        method="identity",  # "identity", "open", "neutral_loss", "hybrid", "all", or list of the above
        precursor_ions_removal_da=0.5,
        noise_threshold=0.0,
        min_ms2_difference_in_da=ms2_tol * 2.2,
        reverse=True
    )

    # score_arr, matched_peak_arr, spec_usage_arr, scaled_score_arr = search_result['identity_search']

    print(search_result)
