# DisulfideBondSiteNN

This project is an implementation of the paper "[Prediction of disulfide bond engineering sites using a machine 
learning method](https://www.nature.com/articles/s41598-020-67230-z)" by Gao et al.

Running this program is pretty simple: first, run the script to extract data from your list of PDB IDs
`MakeDataSet.py <string>path_to_protein_ids`
Then simply run the predictor, and it will handle the training of the model:
`PredictNovelDisulfideBridge.py <string>PDB_prot_ID`
