# GNet4FL  

GNet4FL: A fault localization method based on graph convolutional neural networks to predict potential faulty locations via static and dynamic features of source code.
This repository includes data processing and model constructing.  
  
## Environment
You need to have the following library

- Javalang  https://github.com/c2nes/javalang
- Pytorch V1.60 https://pytorch.org
	> Not sure if other versions of pytorch are supported.
- Numpy
- Scikit-learn

## Organization
The following is the organization of the repository.

```
│  ast_statement.py
│  data_load.py
│  main.py
│  models.py
│  README.md
│  utils.py
│  
├─MS
│  │  model.py
```

+ ast_statement.py :  #Constructing ASTs 
+ data_load.py: #Loading processed data
+ main.py  :#Main file for running the model
+ models.py :  #Mainly GraphSAGE models
+ utils.py  : #Some parameter settings


## Run

 1. Data processing, i.e. run ast_statement.py. Due to data being overwritten, we are unable to provide the dataset. 
 Please collect the defects4j related information to construct the data.
 2. Set the project and version name, e.g.,project='Chart',version=1
 3. run main.py
 
Due to many model parameters and the fact that their initialization is not fixed, the results may differ from our paper. 


