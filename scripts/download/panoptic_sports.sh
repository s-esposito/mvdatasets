# Downlaod the panoptic-sports dataset

# create data folder if it doesn't exist
mkdir -p data

# download from source
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip -O data/pan_sports.zip

# unzip
unzip data/pan_sports.zip -d data/pan_sports

# remove zip file
rm data/pan_sports.zip