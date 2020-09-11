# Download captioning data

# Captions
wget -P ../data/mscoco http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip ../data/mscoco/annotations_trainval2014.zip -d ../data/mscoco
rm ../data/mscoco/annotations_trainval2014.zip

curl https://cloudstor.aarnet.edu.au/plus/s/DiB1OTzcU55Q3VI/download --output ../data/mscoco/caption_datasets.zip
unzip ../data/mscoco/caption_datasets.zip -d ../data/mscoco/caption_datasets
rm ../data/mscoco/caption_datasets.zip
