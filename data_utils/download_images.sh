# General data for captioning and VQA

# Train
wget -P ../data/mscoco http://images.cocodataset.org/zips/train2014.zip
unzip ../data/mscoco/train2014.zip -d ../data/mscoco
rm ../data/mscoco/train2014.zip

# Validation
wget -P ../data/mscoco http://images.cocodataset.org/zips/val2014.zip
unzip ../data/mscoco/val2014.zip -d ../data/mscoco
rm ../data/mscoco/val2014.zip

# Test
wget -P ../data/mscoco http://images.cocodataset.org/zips/test2015.zip
unzip ../data/mscoco/test2015.zip -d ../data/mscoco
rm ../data/mscoco/test2015.zip

# GloVe Vectors
mkdir ../data/glove
wget -P ../data/glove http://nlp.stanford.edu/data/glove.6B.zip
unzip ../data/glove/glove.6B.zip -d ../data/glove
rm ../data/glove/glove.6B.zip

# Pretrained image Features
mkdir ../data/trainval_36
wget -O ../data/trainval_36/trainval_36.zip https://cloudstor.aarnet.edu.au/plus/s/jM0tQlFdtw5thn1/download
unzip ../data/trainval_36/trainval_36.zip -d ../data
rm ../data/trainval_36/trainval_36.zip
