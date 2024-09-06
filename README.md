## Full-scan structure annotation

In mass spectrometry analysis, compounds can be fragmented in- or post-ion source, resulting in fragment ions detected in MS1 spectra.

Here we provide a streamlined workflow to annotate MS1 spectra. This strategy works for both **LC-MS** data and **MS imaging** data.

----------------

### MS1 annotation
Workflow
![Annotation workflow](fig/workflow.png)

Example annotations
![Example annotation](fig/eg_annotation.png)

----------------

### Run the workflow
- Clone the repository.
```bash
 git clone git@github.com:Philipbear/ms1_id.git
```
- Install the dependencies.
```bash
 pip install -r requirements.txt
```
- Run [`ms1id_lcms.py`](https://github.com/Philipbear/ms1_id/blob/main/ms1id_lcms.py) for LC-MS data, and [`ms1id_msi.py`](https://github.com/Philipbear/ms1_id/blob/main/ms1id_msi.py) for MS imaging data.

Indexed libraries are needed for the workflow. You can download the indexed GNPS library [here](https://github.com/Philipbear/ms1_id/releases/tag/v0.0.1).

----------------

### Data
- MS/MS library
  - GNPS: [ALL_GNPS_NO_PROPOGATED.msp](https://external.gnps2.org/gnpslibrary), downloaded on July 17, 2024
    - indexed version [available here](https://github.com/Philipbear/ms1_id/releases/tag/v0.0.1)
  - NIST20 (commercially available)

- LC-MS data
  - Chemical standard data (links: [GNPS/MassIVE MSV000095789](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=361b126b35f64bb89a99e7a9974cf8a7))
  - NIST human feces data (links: [GNPS/MassIVE MSV000095787](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=fa2bf73306ef4e7d89a3e3d3a4cb76d1))
  - IBD data (links: [original paper](https://www.nature.com/articles/s41586-019-1237-9), [data](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Project&ProjectID=PR000639))

- MS imaging data
  - Mouse brain (links: [original paper](https://www.nature.com/articles/nmeth.4072), [data](https://www.ebi.ac.uk/metabolights/editor/MTBLS313))
  - Mouse body (links: [METASPACE dataset](https://metaspace2020.eu/dataset/2022-07-08_20h45m00s))

[//]: # (  - Mouse kidney &#40;links: [METASPACE dataset]&#40;https://metaspace2020.eu/dataset/2019-03-28_18h03m06s&#41;&#41;)
[//]: # (  - Mouse liver &#40;links: [METASPACE dataset]&#40;https://metaspace2020.eu/dataset/2017-02-23_09h51m18s&#41;&#41;)

----------------

### Citation
``
add citation
``