# Downlaod the Mip-NeRF360 dataset

# create data folder if it doesn't exist
mkdir -p data

# download from source
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O data/mipnerf360.zip

# unzip
unzip data/mipnerf360.zip -d data/mipnerf360

# remove zip file
rm data/mipnerf360.zip