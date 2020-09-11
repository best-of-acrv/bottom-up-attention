# Download VQA data

# Questions
mkdir ../data/mscoco
wget -P ../data/mscoco https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip ../data/mscoco/v2_Questions_Train_mscoco.zip -d ../data/mscoco
rm ../data/mscoco/v2_Questions_Train_mscoco.zip

wget -P ../data/mscoco https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip ../data/mscoco/v2_Questions_Val_mscoco.zip -d ../data/mscoco
rm ../data/mscoco/v2_Questions_Val_mscoco.zip

wget -P ../data/mscoco https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip ../data/mscoco/v2_Questions_Test_mscoco.zip -d ../data/mscoco
rm ../data/mscoco/v2_Questions_Test_mscoco.zip

# Annotations
wget -P ../data/mscoco https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip ../data/mscoco/v2_Annotations_Train_mscoco.zip -d ../data/mscoco
rm ../data/mscoco/v2_Annotations_Train_mscoco.zip

wget -P ../data/mscoco https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip ../data/mscoco/v2_Annotations_Val_mscoco.zip -d ../data/mscoco
rm ../data/mscoco/v2_Annotations_Val_mscoco.zip


