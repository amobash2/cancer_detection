# Input data for training, validation and test will have following format:
data  __ train __ image1.tif
     |       |___ image2.tif
     |       |___ ...
     |
     |__ validation __ image1000.tif
     |             |__ ...
     |
     |__ test __ image11000.tif
     |       |__ ...
     |__ all_labels.csv


all_labels.csv containes list of all file names with corresponding correct label. 
* The file contains two columns "id" and "label". 
* Where "id" columns contains image1, image2 and label can be 0 (cancer not detected) and 1 (metastatic cancer is detected).
