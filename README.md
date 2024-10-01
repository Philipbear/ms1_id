# Structure annotation of full-scan MS data
[![Developer](https://img.shields.io/badge/Developer-Shipei_Xing-yellowgreen?logo=github&logoColor=white)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat&logo=apache)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg?style=flat&logo=python&logoColor=lightblue)

Full-scan MS data from both LC-MS and MS imaging capture multiple ion forms, including their in-source fragments. 
Here we leverage such fragments to structurally annotate full-scan data from **LC-MS** or **MS imaging** by matching against MS/MS spectral libraries.


## MS1 annotation
Workflow
![Annotation workflow](fig/workflow.png)

Example annotations
![Example annotation](fig/eg_annotation.png)


## Run the workflow
- Clone the repository.
```bash
 git clone git@github.com:Philipbear/ms1_id.git
```
- Install the dependencies (Python 3.9+ required).
```bash
 pip install -r requirements.txt
```
- Run [`ms1id_lcms.py`](https://github.com/Philipbear/ms1_id/blob/main/ms1id_lcms.py) for LC-MS data, and [`ms1id_msi.py`](https://github.com/Philipbear/ms1_id/blob/main/ms1id_msi.py) for MS imaging data.

Indexed libraries are needed for the workflow. You can download the indexed GNPS library [here](https://github.com/Philipbear/ms1_id/releases/tag/v0.0.1). 
To build your own indexed library, run [`bin/lcms/_reverse_matching.py`](https://github.com/Philipbear/ms1_id/blob/main/bin/lcms/_reverse_matching.py).


## Data
- GNPS MS/MS library
  - [ALL_GNPS_NO_PROPOGATED.msp](https://external.gnps2.org/gnpslibrary), downloaded on July 17, 2024
  - indexed version [available here](https://github.com/Philipbear/ms1_id/releases/tag/v0.0.1)

- LC-MS data
  - Chemical standard data ([GNPS/MassIVE MSV000095789](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=361b126b35f64bb89a99e7a9974cf8a7))
  - NIST human feces data ([GNPS/MassIVE MSV000095787](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=fa2bf73306ef4e7d89a3e3d3a4cb76d1))
  - IBD data ([original paper](https://www.nature.com/articles/s41586-019-1237-9), [data](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Project&ProjectID=PR000639))

- MS imaging data
  - Mouse brain ([original paper](https://www.nature.com/articles/nmeth.4072), [data](https://www.ebi.ac.uk/metabolights/editor/MTBLS313))
  - Mouse body ([METASPACE dataset](https://metaspace2020.eu/dataset/2022-07-08_20h45m00s))


## Citation
``
to be added
``


## License
This project is licensed under the Apache 2.0 License.