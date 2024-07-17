# ms1_id

ISF account for >30% of mass spec signals.
Here we utilize MS1 ISF, to match against low-energy reference MS/MS spectra.
This allows us to provide annotations using MS1 data, which could be expanded to MS imaging.


## methods
use masscube workflow, add feature correlation, do rev cos search against library.

## discussions
- which compound classes are easier to have ISF, such that they can be annotated using MS1 (nist20)
- rev cos match FDR
- MS imaging

## data
- GNPS: ALL_GNPS_NO_PROPOGATED.msp, downloaded on July 17, 2024
- MassBank (https://github.com/MassBank/MassBank-data/releases, MassBank_NIST, 2024.6 release), 120184 spectra
- NIST20
