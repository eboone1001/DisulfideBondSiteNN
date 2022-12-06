import os.path
import sys
import DisulfideBondSiteNN
import torch
from torch import tensor
import PDBWrangler as pdb
import pdb2sql_utils


if __name__ == "__main__":
    """
    This is the source for the actual bond site predictor.  
    """
    argumentList = sys.argv[1:]
    protein_id = argumentList[0][:4]
    disulf_NN = DisulfideBondSiteNN.DisulfideBondSiteNN()

    # Loads the model or trains a new one.
    if os.path.exists(DisulfideBondSiteNN.MODEL_PARAM_FILE):
        disulf_NN.load_state_dict(torch.load(DisulfideBondSiteNN.MODEL_PARAM_FILE))
    else:
        DisulfideBondSiteNN.train(disulf_NN)

    # Downloading and reading the PDB file
    pdb2sql_utils.fetch(protein_id)
    pdb_file = protein_id + ".pdb"
    subj_prot = pdb.Protein(pdb_file)
    os.remove(pdb_file)

    # Pre-processing to remove problem residues and get all alpha carbon locations
    subj_prot.remove_glycine()
    residue_locations = subj_prot.get_residue_loc()
    residues = [pdb.AAResidue(pdb.get_residue_atoms(res_loc, subj_prot.atom_lines)) for res_loc in residue_locations]

    # finding residues that are close enough to be of interest
    residues_of_interest = []
    for i in range(len(residues)):
        for j in range(i, len(residues)):
            if 3 < pdb.ca_dist(residues[i], residues[j]) < 7.5:
                residues_of_interest.append((residues[i], residues[j]))

    # running the neural network to make predictions
    potential_disulfide_sites = []
    for resid_pair in residues_of_interest:
        dist_vec = pdb.residue_dist(resid_pair)
        classification = round(float(disulf_NN(tensor(dist_vec))))
        if classification == 1:
            potential_disulfide_sites.append(resid_pair)

    # Enjoy the predictions!
    print([(res[0].loc, res[1].loc) for res in potential_disulfide_sites])

