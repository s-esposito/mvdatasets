# Downlaod the HyperNeRF dataset

# create data folder if it doesn't exist
mkdir -p data

# download from source

# all scenes
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_americano.zip -O data/americano.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_cross-hands.zip -O data/cross-hands.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_espresso.zip -O data/espresso.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_oven-mitts.zip -O data/oven-mitts.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_split-cookie.zip -O data/split-cookie.zip
wget https://github.com/google/hypernerf/releases/download/v0.1/misc_tamping.zip -O data/tamping.zip

# unzip
unzip data/americano.zip -d data/hypernerf
unzip data/cross-hands.zip -d data/hypernerf
unzip data/espresso.zip -d data/hypernerf
unzip data/oven-mitts.zip -d data/hypernerf
unzip data/split-cookie.zip -d data/hypernerf
unzip data/tamping.zip -d data/hypernerf

# rename split-cookie1 to split-cookie
mv data/hypernerf/cross-hands1 data/hypernerf/cross-hands

# remove zip file
rm data/americano.zip
rm data/cross-hands.zip
rm data/espresso.zip
rm data/oven-mitts.zip
rm data/split-cookie.zip
rm data/tamping.zip
