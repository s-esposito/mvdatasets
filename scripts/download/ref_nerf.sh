# Downlaod the Ref-NeRF dataset

# create data folder if it doesn't exist
mkdir -p data

# download from source
wget https://storage.googleapis.com/gresearch/refraw360/ref.zip -O data/ref_nerf.zip

# unzip
unzip data/ref_nerf.zip -d data

# remove zip file
rm data/ref_nerf.zip