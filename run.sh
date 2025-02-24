# This script downloads indexed GNPS libraries, a public LC-MS dataset (MSV000095868) via FTP, and runs ms1_id
# Requirements: conda, lftp, wget, unzip

# install ms1_id
conda create -n ms1_id python=3.10
conda activate ms1_id
pip install ms1_id

# download indexed libraries
wget https://github.com/Philipbear/ms1_id/releases/latest/download/indexed_gnps_libs.zip

# unzip libraries
unzip indexed_gnps_libs.zip -d db

# download data
lftp ftp://massive.ucsd.edu/v08/MSV000095868/peak/Lipids_pos
mirror . ./MSV000095868/data

# run ms1_id
ms1_id lcms --project_dir MSV000095868 --sample_dir data --ms1_id_libs db/gnps.pkl db/gnps_k10.pkl --ms2_id_lib db/gnps.pkl
