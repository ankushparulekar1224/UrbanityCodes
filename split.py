import splitfolders
input_file = "C:/urbanity/totalData"
output_file ="C:/urbanity/FinalData"
 
splitfolders.ratio(input_file, output_file,seed=1337, ratio  =(.8,.1,.1),group_prefix= None)