# ms1_id

In high-resolution mass spectrometry analysis, compounds can be fragmented in or post ion source.

ISF account for >30% of mass spec signals.
Here we utilize MS1 ISF, to match against low-energy reference MS/MS spectra.
This allows us to provide annotations using MS1 data, which could be expanded to MS imaging.

-------------------------------------

## methods
use masscube workflow, add feature correlation, do rev cos search against library.

## discussions
- some compound classes are easier to have ISF, such that they can be annotated using MS1
- MS imaging can have better correlation approaches, ML (CNN, etc)

## data
- MS/MS library
  - GNPS: ALL_GNPS_NO_PROPOGATED.msp, downloaded on July 17, 2024
  - NIST20 (commercially available)


- MS imaging data
  - MTBLS313
  - mouse body data: https://metaspace2020.eu/dataset/2022-07-08_20h45m00s


