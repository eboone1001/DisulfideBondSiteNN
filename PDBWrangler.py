import math

import numpy as np
import re
from scipy.spatial import distance


def parse_location(atom_line):
    """
    Takes PBD format ATOM line and converts into numpy array of floats indicating location
    :param atom_line:
    :return: Array of floats indicating XYZ position of atom
    """
    line_data = atom_line.split()
    float_data = [float(string) for string in line_data[6:9]]
    return np.array(float_data)


def get_bond_loc(ssbond_line):
    """
    :param ssbond_line:
    :return: Tuple pairs of "Subunit, Sequence Position" format indicating pairs of amino acids
    """
    pos_bond_loc = []
    neg_bond_loc = []
    for line in ssbond_line:
        func_group = line[15]
        first_loc = int(line[16:21])
        second_loc = int(line[30:35])

        pos_bond_loc.append((func_group + str(first_loc).rjust(4),
                         func_group + str(second_loc).rjust(4)))

        neg_bond_loc.append((func_group + str((first_loc + 1)).rjust(4),
                         func_group + str((first_loc - 1)).rjust(4),
                         func_group + str((second_loc + 1)).rjust(4),
                         func_group + str((second_loc - 1)).rjust(4)))

    return pos_bond_loc, neg_bond_loc


def get_residue_atoms(location, atom_lines):
    """
    :param location:
    :param atom_lines:
    :return: ATOM lines corresponding to the amino acid specified in "location"
    """
    return [line for line in atom_lines if location in line]


def residue_dist(residue_pairs: tuple):
    """
    Given two residue objects, calculates 10x10 vector of euclidian distances, reshaped into a 45 member vector to
    remove redundant data.
    :param residue_pairs:
    :return: 45 member vector of euclidian distances
    """
    resid1 = residue_pairs[0]
    resid2 = residue_pairs[1]
    atom_loc = np.concatenate((resid1.atom_loc, resid2.atom_loc))
    num_atoms = len(atom_loc)
    if num_atoms != 10:
        raise IOError("Location data should be 10x3 not " + str(num_atoms) + "x" + str(len(atom_loc[0])))

    distances = []

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distances.append(distance.euclidean(atom_loc[i], atom_loc[j]))

    """if len(distances) != 45:
        raise IOError("Output vector is incorrect length:" + str(len(distances)))"""

    return np.array(distances)


class AAResidue:
    """
    A class for easily extracting, storing, and accessing the locations of the atoms in a given AA residue
    """
    def __init__(self, atom_lines):
        # Note for future Eric: take this check out if reusing this code in a later project
        if len(atom_lines) == 0:
            print("STOP HERE")
        # raise IOError("Too many atom lines being passed to residue init")

        self.loc = atom_lines[0][17:26]
        # Eventually a 5X3 array
        # Should be in order N, CA, C, O, CB
        self.atom_loc = np.array([parse_location(line) for line in atom_lines[:5]])

        # TODO: Delete this if you end up not using it
        self.N = self.atom_loc[0]
        self.CA = self.atom_loc[1]
        self.C = self.atom_loc[2]
        self.O = self.atom_loc[3]
        # Glycine causes trouble because it has no beta-carbon. This if-statement automatically disqualifies it from
        # being selected as a negative example; the next best fit residue will be picked.
        # TODO: This was giving me trouble for a bit but then "magically" starting working. If NN not working start here
        # TODO: and check to make sure that GLY residues are not making it past the select_neg_pairs() step.
        if "GLY" in atom_lines[0]:
            self.CB = None
        else:
            self.CB = self.atom_loc[4]


def ca_dist(resid1: AAResidue, resid2: AAResidue):
    return distance.euclidean(resid1.CA, resid2.CA)


class Protein:

    def make_residue_pairs(self, residue_locs: tuple):
        """
        Given tuple of two locations of interest (format "Subunit Location" e.g. "B 124") creates two AAResidue opjects
        storing the ATOM location data.  See AAResidue class for more info.
        :param residue_locs: tuple
        :return: tuple of AAResidue objects
        """
        return (AAResidue(get_residue_atoms(residue_locs[0], self.atom_lines)),
                AAResidue(get_residue_atoms(residue_locs[1], self.atom_lines)))

    def __init__(self, source_file_path, chain=""):
        """
        This protein object abstracts the process of extracting the feature vector used in the NN. Simply initialize
        with the path to a PDB file of interest, and the object will store the positive and negative example feature
        vectors in pos_distance_vector and neg_distance_vector respectively.
        :param source_file_path:
        """
        # TODO: Some class variables are large and not needed afterwards: remove if we are having memory issues
        sourcefile = open(source_file_path, "r")
        lines = sourcefile.readlines()
        sourcefile.close()

        self.ssbond_lines = [line for line in lines if
                             "SSBOND" in line and
                             (chain in line[15:17] or chain == "")]

        # This is not working, need to find a way to stop including CG OG1 etc.
        # Look to see if you can find another substring checking function.
        atom_tags = ["N", "CA", "C", "O", "CB"]
        self.atom_lines = [line for line in lines if
                           ("ATOM" in line) and
                           chain in line[21:23] and
                           any([re.search(r'\b%s\b' % tag, line[13:15]) is not None for tag in atom_tags])]

        """
        ("N " or "CA" or "C " or "O " or "CB" == line[13:15]) and
        ("CG" or "CD" != line[13:15])]"""

        pos_loc, neg_loc = get_bond_loc(self.ssbond_lines)

        self.positive_bonds = [self.make_residue_pairs(loc) for loc in pos_loc]
        self.negative_bonds = []
        self.select_neg_pairs(neg_loc)

        self.pos_distance_vectors = [residue_dist(resid_pair) for resid_pair in self.positive_bonds]
        self.neg_distance_vectors = [residue_dist(resid_pair) for resid_pair in self.negative_bonds]

    def remove_glycine(self):
        temp = [line for line in self.atom_lines if ("GLY" not in line)]
        self.atom_lines = temp

    def get_residue_loc(self):
        return [line[21:26] for line in self.atom_lines if re.search(r'\b%s\b' % "N", line[13:15])]

    def select_neg_pairs(self, neg_loc):
        """
        Original paper calls for selecting a negative example from the residues surrounding a sulfur bridge pair.
        See the original paper (linked in Readme) for exhaustive explanation of the algorithm.
        :param neg_loc:
        :return:
        """
        for loc_set in neg_loc:
            residues = [AAResidue(get_residue_atoms(loc, self.atom_lines)) for loc in loc_set]
            ind1 = -1
            ind2 = -1
            min = math.inf
            for i in range(4):
                for j in range(i+1,4):
                    residue1 = residues[i]
                    residue2 = residues[j]
                    if residue1.CB is None or residue2.CB is None:
                        continue
                    dist = distance.euclidean(residue1.CA, residue2.CA)
                    if dist < min:
                        if i == j: raise IOError("Closest CA neighbor is itself: debug")

                        ind1 = i
                        ind2 = j
                        min = dist

            self.negative_bonds.append((residues[ind1], residues[ind2]))


# For developer testing: maybe remove. Command line interface not necessary.
if __name__ == "__main__":
    ex_prot = Protein("testPDBfiles/pdb6lr4.ent")
    """for line in ex_prot.atom_lines:
        if (("ATOM" in line) and
            ("N " or "CA" or "C " or "O " or "CB" == line[13:15]) and
            ("CG" or "CD" != line[13:15])):
            print(line)"""
    print(ex_prot.ssbond_lines)
    print(ex_prot.atom_lines)
    print(get_bond_loc(ex_prot.ssbond_lines))

    print(ex_prot.positive_bonds)
    print(ex_prot.negative_bonds)
    print(ex_prot.pos_distance_vectors)
    print(ex_prot.neg_distance_vectors)


