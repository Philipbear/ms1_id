# ms1_id

ISF account for >30% of mass spec signals.
Here we utilize MS1 ISF, to match against low-energy reference MS/MS spectra.
This allows us to provide annotations using MS1 data, which could be expanded to MS imaging.


## methods
use masscube workflow, add feature correlation, do rev cos search against library.

## discussions
- some compound classes are easier to have ISF, such that they can be annotated using MS1
- MS imaging can have better correlation approaches, ML (CNN, etc)

## data
- MS/MS library
  - GNPS: ALL_GNPS_NO_PROPOGATED.msp, downloaded on July 17, 2024
  - NIST20


- MS imaging data
  - MTBLS313
  - mouse body data 1: https://metaspace2020.eu/dataset/2017-05-17_19h49m04s
  - mouse body data 2: https://metaspace2020.eu/dataset/2022-07-08_20h45m00s


