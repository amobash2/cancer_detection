### Input data for training, validation and test will have following format:
data/  
* train/
     * image1.tif
     * image2.tif
     * ...
* validation/
     * image1000.tif
     * ...
* test/ 
     * image11000.tif
     * ...
* all_labels.csv


all_labels.csv containes list of all file names with corresponding correct label. 
* The file contains two columns "id" and "label". 
* Where "id" columns contains image1, image2 and label can be 0 (cancer not detected) or 1 (metastatic cancer is detected).
