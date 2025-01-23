# Downlaod the iphone (shape-of-motion) dataset

# create data folder if it doesn't exist
mkdir -p data

# download from google drive
gdown 1QJQnVw_szoy_k5x9k_BAE2BWtf_BXqKn -O data/apple.zip
gdown 1QihG5A7c_bpkse5b0OBdqqFThgX0kDyZ -O data/backpack.zip
gdown 1b9Y-hUm9Cviuq-fl7gG-q7rUK7j0u2Rv -O data/block.zip
gdown 1inkHp24an1TyWvBekBxu2wRIyLQ0gkhO -O data/creeper.zip
gdown 1frv8miU24Dl7fqblYt7zkwj129ci-68U -O data/handwavy.zip
gdown 1BmuxJXKi6dVaNOjmppuETQsaspAV9Wca -O data/haru-sit.zip
gdown 1OpgF2ILf43jcN-226wQcxImjcfMAVOwA -O data/mochi-high-five.zip
gdown 15PirJRqsT5lLjuGdLWALBDFMQanj8FTh -O data/paper-windmill.zip
gdown 1Uc2BXpONnWhxKNs6tKMle0MiSVMVZsuB -O data/pillow.zip
gdown 1055wcQk-ZfVWXa_g-dpQIRQy-kLBL_Lk -O data/spin.zip
gdown 18sjQQMU6AijyXg4BoucLX82R959BYAzz -O data/sriracha-tree.zip
gdown 1Mqm4C1Oitv4AsDM2n0Ojbt5pmF_qXVfI -O data/teddy.zip

# unzip
unzip -o data/apple.zip -d data/iphone_som
unzip -o data/backpack.zip -d data/iphone_som
unzip -o data/block.zip -d data/iphone_som
unzip -o data/creeper.zip -d data/iphone_som
unzip -o data/handwavy.zip -d data/iphone_som
unzip -o data/haru-sit.zip -d data/iphone_som
unzip -o data/mochi-high-five.zip -d data/iphone_som
unzip -o data/paper-windmill.zip -d data/iphone_som
unzip -o data/pillow.zip -d data/iphone_som
unzip -o data/spin.zip -d data/iphone_som
unzip -o data/sriracha-tree.zip -d data/iphone_som
unzip -o data/teddy.zip -d data/iphone_som

# remove zip file
rm data/apple.zip
rm data/backpack.zip
rm data/block.zip
rm data/creeper.zip
rm data/handwavy.zip
rm data/haru-sit.zip
rm data/mochi-high-five.zip
rm data/paper-windmill.zip
rm data/pillow.zip
rm data/spin.zip
rm data/sriracha-tree.zip
rm data/teddy.zip



