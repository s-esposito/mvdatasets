# Downlaod the nerf_llff dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 11PhkBXZZNYTD2emdG1awALlhCnkq7aN- -O data/nerf_llff.zip

# unzip
unzip data/nerf_llff.zip -d data

# rename nerf_llff to llff
mv data/nerf_llff_data data/nerf_llff

# remove zip file
rm data/nerf_llff.zip