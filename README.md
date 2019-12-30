# SubmitionINF554

## Text processing
Python scripts to run in order :
1. Text_Processing.py : Input : all texts files at './data/node_information/text/', Output : dict of process texts at './node_info_snl.json'
2. Build_dict.py : Input : './node_info_snl.json', Output : Dictionary at './reduced_dict.dict'
3. Build_bow.py : Input : './node_info_snl.json' and './reduced_dict.dict', Output : './BOW_pca.json'
4. nn_bow.py : Input :'./BOW_pca.json', Output : './predictionBOW.csv'
