import os

import PDBWrangler
import pdb2sql_utils as utils
import os
import sys
import numpy as np


def gen_feat_vectors(path_to_pdb_file):
    """
    Helper function to grab feature vectors from where they are stored in the protein.
    :param path_to_pdb_file:
    :return: tuple of lists of feature vectors
    """
    protein = PDBWrangler.Protein(path_to_pdb_file)
    return protein.pos_distance_vectors, protein.neg_distance_vectors


def make_data_set_train(path_to_protein_ids):
    """
    Takes file of PBD protein IDs and retrieves the data file from the server, and parses it into a Protein object, then
    saves examples to a file.
    :param path_to_protein_ids:
    """

    # Parsing PDB IDs
    if not os.path.isdir("./PDBfiles"):
        os.mkdir("./PDBfiles")
    prot_id_file = open(path_to_protein_ids, "r")
    lines = prot_id_file.readlines()
    lines = lines[4:]
    prot_id_file.close()

    # Writing examples to a file
    outfile = open("feature_vectors.txt", "a")
    np.set_printoptions(linewidth=np.inf)
    for line in lines:
        ids = line.split()
        for pdb_id in ids:
            utils.fetch(pdb_id[:4], outdir="./PDBfiles")
            try:
                prot = PDBWrangler.Protein("./PDBfiles/" + pdb_id[:4] + ".pdb", pdb_id[4])
            except:
                os.remove("./PDBfiles/" + pdb_id[:4] + ".pdb")
                continue
            for vector in prot.pos_distance_vectors:
                outfile.write("1\t"+str(vector)[1:-1]+"\n")
            for vector in prot.neg_distance_vectors:
                outfile.write("0\t"+str(vector)[1:-1]+"\n")
            os.remove("./PDBfiles/" + pdb_id[:4] + ".pdb")

    os.rmdir("PDBfiles")


if __name__ == "__main__":
    argumentList = sys.argv[1:]
    path_to_protein_ids = argumentList[0][:4]
    make_data_set_train(path_to_protein_ids)
