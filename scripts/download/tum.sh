# Downlaod the TUM dataset

# create data folder if it doesn't exist
mkdir -p data/tum

wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz -O data/tum/rgbd_dataset_freiburg1_desk.tgz
wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz -O data/tum/rgbd_dataset_freiburg2_xyz.tgz
wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -O data/tum/rgbd_dataset_freiburg3_long_office_household.tgz

# unzip
tar -xvzf data/tum/rgbd_dataset_freiburg1_desk.tgz -C data/tum
tar -xvzf data/tum/rgbd_dataset_freiburg2_xyz.tgz -C data/tum
tar -xvzf data/tum/rgbd_dataset_freiburg3_long_office_household.tgz -C data/tum

# remove zip file
rm data/tum/rgbd_dataset_freiburg1_desk.tgz
rm data/tum/rgbd_dataset_freiburg2_xyz.tgz
rm data/tum/rgbd_dataset_freiburg3_long_office_household.tgz