# This script downloads indexed GNPS libraries, a public LC-MS dataset (MSV000095868) via FTP, and runs ms1_id
# Requirements: conda, lftp, wget, unzip

# install ms1_id
conda create -n ms1_id python=3.10
conda activate ms1_id
pip install ms1_id

# download indexed libraries
wget https://github.com/Philipbear/ms1_id/releases/latest/download/gnps.zip

# unzip libraries
unzip gnps.zip -d db

# download data
lftp ftp://massive.ucsd.edu/v08/MSV000095868/peak/Lipids_pos
mirror . ./MSV000095868/data

# run ms1_id
ms1_id lcms --project_dir MSV000095868 --sample_dir data --ms1_id_libs db/gnps.pkl db/gnps_k10.pkl --ms2_id_lib db/gnps.pkl

##################
## MS imaging data
# ms1_id msi -i spotted_stds --libs ../data/gnps_minmz100.pkl ../data/gnps_minmz100_k10.pkl -m neg
# ms1_id msi -i mouse_body --libs ../data/gnps_minmz100.pkl ../data/gnps_minmz100_k10.pkl --sn_factor 5.0 -m pos
# ms1_id msi -i mouse_brain --libs ../data/gnps_minmz100.pkl ../data/gnps_minmz100_k10.pkl --sn_factor 3.0 -m pos
# ms1_id msi -i plant_root --libs ../data/gnps.pkl ../data/gnps_k10.pkl --min_pixel_overlap 10 --sn_factor 0.0 -m neg --mz_ppm_tol 10.0
# ms1_id msi -i human_liver --libs ../data/gnps_minmz200.pkl ../data/gnps_minmz200_k10.pkl --sn_factor 3.0 -m pos --mz_ppm_tol 10.0
# ms1_id msi -i hepatocytes --libs ../data/gnps_minmz200.pkl ../data/gnps_minmz200_k10.pkl --sn_factor 3.0 -m pos
# ms1_id msi -i hela --libs ../data/gnps_minmz200.pkl ../data/gnps_minmz200_k10.pkl --sn_factor 3.0 -m neg
# ms1_id msi -i mouse_kidney --libs ../data/gnps_minmz100.pkl ../data/gnps_minmz100_k10.pkl --sn_factor 10.0 -m pos
# ms1_id msi -i mouse_brain_malditof --libs ../data/gnps_minmz100.pkl ../data/gnps_minmz100_k10.pkl --sn_factor 3.0 -m pos --mz_ppm_tol 50.0
# ms1_id msi -i human_kidney --libs ../data/gnps_minmz300.pkl ../data/gnps_minmz300_k10.pkl --sn_factor 3.0 -m pos  --mz_ppm_tol 10