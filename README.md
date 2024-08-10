## MS1-based structural annotation

In mass spectrometry analysis, compounds can be fragmented in- or post-ion source, resulting in fragment ions detected in MS1 spectra.

Here we utilize MS1 data to match against reference MS/MS spectra, to provide structural annotations.

This strategy works for both **LC-MS** data and **MS imaging** data.

### Run the workflow
- Clone the repository.
```bash
 git clone git@github.com:Philipbear/ms1_id.git
```
- Install the dependencies.
```bash
 pip install -r requirements.txt
```
- Run `ms1id_lcms.py` for LC-MS data, and `ms1id_msi.py` for MS imaging data.


### Data
- MS/MS library
  - GNPS: [ALL_GNPS_NO_PROPOGATED.msp](https://external.gnps2.org/gnpslibrary), downloaded on July 17, 2024
    - indexed version [available here](https://github.com/Philipbear/ms1_id/releases/tag/v0.0.1)
  - NIST20 (commercially available)

- LC-MS data
  - NIST human feces data (to be uploaded)
  - IBD data ([original paper](https://www.nature.com/articles/s41586-019-1237-9), [data](https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Project&ProjectID=PR000639))

- MS imaging data
  - Mouse brain ([original paper](https://www.nature.com/articles/nmeth.4072), [data](https://www.ebi.ac.uk/metabolights/editor/MTBLS313))
  - Mouse body ([METASPACE link](https://metaspace2020.eu/dataset/2022-07-08_20h45m00s))


-------------------------------------

### methods
use masscube workflow, add feature correlation, do rev cos search against library.

### discussions
- some compound classes are easier to have ISF, such that they can be annotated using MS1
- MS imaging can have better correlation approaches, ML (CNN, etc)




