# Downlaod the nerf_furry dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 18W9aSIL4SnCDaHJ8uaGvWF4GgAze14lm -O data/nerf_furry.zip

# unzip
unzip data/nerf_furry.zip -d data

# remove zip file
rm data/nerf_furry.zip