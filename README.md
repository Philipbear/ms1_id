# ms1_id
[![Developer](https://img.shields.io/badge/Developer-Shipei_Xing-orange?logo=github&logoColor=white)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
[![PyPI](https://img.shields.io/pypi/v/ms1_id?color=green)](https://pypi.org/project/ms1_id/)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat&logo=apache)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg?style=flat&logo=python&logoColor=lightblue)

Full-scan MS data from both LC-MS and MS imaging capture multiple ion forms, including their in/post-source fragments. 
Here we leverage such fragments to structurally annotate full-scan data from **LC-MS** or **MS imaging** by matching against MS/MS spectral libraries.

`ms1_id` is a Python package that annotates full-scan MS data using tandem MS libraries, specifically:
- annotate pseudo MS/MS spectra: **mgf** files
- annotate LC-MS data: **mzML** or **mzXML** files
- annotate MS imaging data: **imzML** and **ibd** files
- build indexed MS/MS libraries from **mgf** or **msp** files (see [Flash entropy](https://github.com/YuanyueLi/FlashEntropySearch) for more details)

#### Workflow
![Annotation workflow](fig/workflow.png)


#### Example annotations
![Example annotation](fig/eg_annotation.png)

## Installation
```bash
pip install ms1_id
```
Python 3.9+ is required. It has been tested on macOS (14.6, M2 Max) and Linux (Ubuntu 20.04).


## Usage

Note: Indexed libraries are needed for the workflow. You can download the indexed GNPS library [here](https://github.com/Philipbear/ms1_id/releases).
```bash
# For LC-MS data
wget https://github.com/Philipbear/ms1_id/releases/latest/download/gnps.zip
unzip gnps.zip -d db

# For MS imaging data (fragments with mz < 100 are removed, as they are not usually included in MS imaging data)
wget https://github.com/Philipbear/ms1_id/releases/latest/download/gnps_minmz100.zip
unzip gnps_minmz100.zip -d db
```

---------

### Annotate pseudo MS/MS spectra
If you have pseudo MS/MS spectra in **mgf** format, you can directly annotate them:
  ```bash
  ms1_id annotate --input_file pseudo_msms.mgf --libs db/gnps.pkl db/gnps_k10.pkl --min_score 0.7 --min_matched_peak 3
  ```
Here, two indexed libraries are searched against, and the result tsv files will be saved in the same directory as the input file.

For more options, run:
  ```bash
  ms1_id annotate --help
  ```

---------

### Annotate LC-MS data
To annotate LC-MS data, here is an example command:
  ```bash
  ms1_id lcms --project_dir lc_ms --sample_dir data --ms1_id_libs db/gnps.pkl db/gnps_k10.pkl --ms2_id_lib db/gnps.pkl
  ```
Here, `lc_ms` is the project directory. Raw mzML or mzXML files are stored in the `lc_ms/data` folder. Both MS1 and MS/MS annotations will be performed. For MS1 annotation, both gnps.pkl and gnps_k10.pkl libraries are used. For MS/MS annotation, the gnps.pkl library is used. Results can be accessed from `aligned_feature_table.tsv`.

For more options, run:
  ```bash
  ms1_id lcms --help
  ```
Expected runtime is ~5-7 min for a single LC-MS file. If it takes longer than 10 min, please increase the `--mass_detect_int_tol` parameter (default: 2e5 for Orbitraps, 5e2 for QTOFs).

---------

### Annotate MS imaging data
To annotate MS imaging data, here is an example command:
  ```bash
  ms1_id msi --input_dir msi --libs db/gnps_minmz100.pkl db/gnps_minmz100_k10.pkl --n_cores 12
  ```
Here, `msi` is the input directory consisting of the imzML and ibd files. All the imzML files in the directory will be annotated individually.
Two libraries are used simultaneously, and 12 cores will be used for parallel processing. Annotation results can be accessed from `ms1_id_annotations_derep.tsv`

For more options, run:
  ```bash
  ms1_id msi --help
  ```
Expected runtime is ~3-20 min for a single MS imaging dataset if at least 12 cores are available.

---------

### Build indexed MS/MS libraries
To build your own indexed library, run:
  ```bash
  ms1_id index --ms2db library.msp --peak_scale_k 10 --peak_intensity_power 0.5
  ```

For more options, run:
  ```bash
  ms1_id index --help
  ```

## Demo
We provide [a demo script](https://github.com/Philipbear/ms1_id/blob/main/run.sh) to prepare the environment, download libraries, download LC-MS data and run the annotation workflow. 
```bash
bash run.sh
```


## Citation
> Shipei Xing, Vincent Charron-Lamoureux, Yasin El Abiead, Huaxu Yu, Oliver Fiehn, Theodore Alexandrov, Pieter C. Dorrestein. Annotating full-scan MS data using tandem MS libraries. [bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.10.14.618269v1).


## Data
| Data type  |                  Dataset                  |                                                  Link                                                   |   Instrument   |
|:----------:|:-----------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:--------------:|
|   LC-MS    |         Pooled chemical standards         |              [MSV000095789](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000095789)               |   Q Exactive   |
|   LC-MS    |             NIST human feces              |              [MSV000095787](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000095787)               |   Q Exactive   |
|   LC-MS    |                IBD dataset                | [PR000639](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Project&ProjectID=PR000639) |   Q Exactive   |
|   LC-MS    |         Mouse feces (lipidomics)          |              [MSV000095868](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000095868)               |     Q-TOF      |
|   LC-MS    |       Komagataella phaffii (yeast)        |              [MSV000090053](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000090053)               |   Q Exactive   |
|   LC-MS    |            Bacterial isolates             |              [MSV000085024](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000085024)               |   Q Exactive   |
|   LC-MS    | Odontotaenius disjunctus microbe isolates |              [MSV000090030](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000090030)               |   Q Exactive   |
|   LC-MS    |       Environmental fungal strains        |              [MSV000090000](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000090000)               |   Q Exactive   |
|   LC-MS    |               Sea water DOM               |              [MSV000094338](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000094338)               |   Q Exactive   |
|   LC-MS    |                 Foam DOM                  |              [MSV000083888](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000083888)               |   Q Exactive   |
|   LC-MS    |                 Ocean DOM                 |              [MSV000083632](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000083632)               |   Q Exactive   |
|   LC-MS    |              Plant extracts               |              [MSV000090975](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000090975)               |   Q Exactive   |
|   LC-MS    |             32 plant species              |              [MSV000090968](https://massive.ucsd.edu/ProteoSAFe/QueryMSV?id=MSV000090968)               |   Q Exactive   |
| MS imaging |    Mouse liver with spotted standards     |                   [METASPACE](https://metaspace2020.org/dataset/2020-12-07_03h16m14s)                   | MALDI-Orbitrap |
| MS imaging |                Mouse brain                |                     [MTBLS313](https://www.ebi.ac.uk/metabolights/editor/MTBLS313)                      |  MALDI-FTICR   |
| MS imaging |                Mouse body                 |                   [METASPACE](https://metaspace2020.eu/dataset/2022-07-08_20h45m00s)                    |  MALDI-FTICR   |
| MS imaging |                Hepatocytes                |                [METASPACE project](https://metaspace2020.eu/project/Rappez_2021_SpaceM)                 | MALDI-Orbitrap |
| MS imaging |         Populus trichocarpa root          |                   [METASPACE](https://metaspace2020.org/dataset/2025-01-07_19h33m53s)                   | MALDI-timsTOF  |
| MS imaging |                Human liver                |                   [METASPACE](https://metaspace2020.org/dataset/2017-09-07_15h14m40s)                   |   MALDI-TOF    |
| MS imaging |               Human kidney                |                   [METASPACE](https://metaspace2020.org/dataset/2024-09-19_00h01m48s)                   | MALDI-timsTOF  |
| MS imaging |               Mouse kidney                |                   [METASPACE](https://metaspace2020.org/dataset/2019-03-28_18h03m06s)                   |  MALDI-FTICR   |
| MS imaging |             Mouse brain (TOF)             |                   [METASPACE](https://metaspace2020.org/dataset/2024-12-21_10h17m55s)                   |   MALDI-TOF    |



## License
This project is licensed under the Apache 2.0 License (Copyright 2024 Shipei Xing).