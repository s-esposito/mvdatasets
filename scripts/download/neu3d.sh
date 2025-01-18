# Downlaod the Neu3D dataset

# create data folder if it doesn't exist
mkdir -p data

# download from source

# all scenes
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip -O data/coffee_martini.zip
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip -O data/cook_spinach.zip
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cut_roasted_beef.zip -O data/cut_roasted_beef.zip
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip -O data/flame_steak.zip
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/sear_steak.zip -O data/sear_steak.zip

# flame salmon
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z01 -O data/flame_salmon_1_split.z01
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z02 -O data/flame_salmon_1_split.z02
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z03 -O data/flame_salmon_1_split.z03
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.zip -O data/flame_salmon_1_split.zip
zip -F data/flame_salmon_1_split.zip --out flame_salmon.zip

# unzip
unzip data/coffee_martini.zip -d data/neu3d
unzip data/cook_spinach.zip -d data/neu3d
unzip data/cut_roasted_beef.zip -d data/neu3d
unzip data/flame_steak.zip -d data/neu3d
unzip data/sear_steak.zip -d data/neu3d
unzip flame_salmon.zip -d data/neu3d

# rename flame_salmon_1 to flame_salmon
mv data/neu3d/flame_salmon_1 data/neu3d/flame_salmon

# remove zip file
rm data/coffee_martini.zip
rm data/cook_spinach.zip
rm data/cut_roasted_beef.zip
rm data/flame_steak.zip
rm data/sear_steak.zip
rm data/flame_salmon_1_split.z01
rm data/flame_salmon_1_split.z02
rm data/flame_salmon_1_split.z03
rm data/flame_salmon_1_split.zip
rm flame_salmon.zip
