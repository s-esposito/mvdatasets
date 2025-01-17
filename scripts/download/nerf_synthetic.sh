# Downlaod the nerf_synthetic dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 1OsiBs2udl32-1CqTXCitmov4NQCYdA9g -O data/nerf_synthetic.zip

# unzip
unzip data/nerf_synthetic.zip -d data

# remove zip file
rm data/nerf_synthetic.zip