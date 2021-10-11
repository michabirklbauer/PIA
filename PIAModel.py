#!/usr/bin/env python3

# SCORING & SCORING HELPER FUNCTIONS - MODEL INTERFACE
# 2021 (c) Micha Johannes Birklbauer
# https://github.com/michabirklbauer/
# micha.birklbauer@gmail.com

version = "1.0.0"
date = "20211007"

"""
DESCRIPTION
PIAModel is an extension to PIAScore and implements a model interface to save
scoring data, parameters and thresholds to apply them to new data.
"""

import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from itertools import islice
import PIAScore
from PIA import PIA
from PIA import Preparation
from PIA import Scoring
from PIA import Comparison
from PIA import exclusion_list
from plip.structure.preparation import PDBComplex

class PIAModel:
    """
    -- DESCRIPTION --
    """

    positives = None
    negatives = None
    strategy = None
    cutoff = None
    train_results = None

    # constructor - initialize a scoring model
    def __init__(self,
                 positives = None,
                 negatives = None,
                 strategy = None,
                 cutoff = None,
                 filename = None):

        """
        -- DESCRIPTION --
        """

        # read values from file if specified
        if filename is not None:
            with open(filename, "r", encoding = "utf-8") as f:
                data = json.load(f)
                f.close()
            self.positives = data["positives"]
            self.negatives = data["negatives"]
            self.strategy = data["strategy"]
            self.cutoff = int(data["cutoff"])
            self.train_results = data["results"]
        else:
            self.positives = positives
            self.negatives = negatives
            self.strategy = strategy
            self.cutoff = cutoff

    # train model (optional)
    def train(self,
              pdb_base_structure,
              sdf_file_1,
              sdf_file_2 = None,
              plot_prefix = None,
              keep_files = False,
              verbose = 1):

        """
        -- DESCRIPTION --
        """

        # create necessary directories
        structures_directory = "piamodel_structures_tmp"
        structures_path = os.path.join(os.getcwd(), structures_directory)
        os.mkdir(structures_path)

        # read files
        filename = sdf_file_1
        if sdf_file_2 is not None:
            # read sdf_file_1 as actives
            with open(sdf_file_1, "r") as f:
                actives = f.read()
                f.close()
            # read sdf_file_2 as inactives
            with open(sdf_file_2, "r") as f:
                inactives = f.read()
                f.close()
            # combine actives and inactives
            content = actives + inactives
            # create combo file
            with open("piamodel_combo_tmp.sdf", "w") as f:
                f.write(content)
                f.close()
            # set filename
            filename = "piamodel_combo_tmp.sdf"

        # preprocessing
        p = Preparation()
        pdb = p.remove_ligands(pdb_base_structure, pdb_base_structure + "_cleaned.pdb")
        ligands = p.get_ligands(filename)
        sdf_metainfo = p.get_sdf_metainfo(filename)
        ligand_names = sdf_metainfo["names"]
        structures = p.add_ligands_multi(pdb_base_structure + "_cleaned.pdb", "piamodel_structures_tmp", ligands)
        # PIA
        result = PIA(structures, ligand_names = ligand_names, poses = "best", path = "current")
        # scoring preparation
        s = Scoring(result.pdb_entry_results, result.best_pdb_entries)
        # generate datasets
        df_train, df_val, df_test = s.generate_datasets(train_output = "piamodel_data_train_tmp.csv",
                                                        val_output = "piamodel_data_val_tmp.csv",
                                                        test_output = "piamodel_data_test_tmp.csv")
        # compare actives and inactives
        ia_info = s.get_actives_inactives()
        comp_train = s.compare(partition = "train")
        comp_val = s.compare(partition = "val")
        comp_test = s.compare(partition = "test")
        # get feature names and information
        features = s.get_feature_information(filename = "piamodel_features_tmp.csv")
        # plot comparisons if plot_prefix is specified
        if plot_prefix is not None:
            comp_train.plot("Actives vs. Inactives - Training Set", filename = plot_prefix + "_comparison_train.png")
            comp_val.plot("Actives vs. Inactives - Validation Set", filename = plot_prefix + "_comparison_val.png")
            comp_test.plot("Actives vs. Inactives - Test Set", filename = plot_prefix + "_comparison_test.png")

        # scoring
        # get optimal features
        if verbose:
            print("Trying to find optimal features...")
        optimized_values_raw = PIAScore.get_optimized_feature_thresholds(features, data_train, data_val)
        opt_key, opt_value = list(islice(optimized_values_raw["ACC"].items(), 1))[0]
        diff_threshold = float(opt_key.split(",")[0].strip())
        active_threshold = float(opt_key.split(",")[1].strip())
        inactive_threshold = float(opt_key.split(",")[2].split(":")[0].strip())
        strat = opt_key.split(":")[-1].strip()
        # filter features
        features_filtered = PIAScore.get_relevant_features(features, diff_threshold, active_threshold, inactive_threshold)
        if verbose:
            print("Finished optimal feature calculation!")
        # get feature impact
        self.positives, self.negatives = PIAScore.get_feature_impact(features_filtered)
        # prepare training, validation and test data
        ## make data copies
        train_result_strat1 = data_train.copy()
        train_result_strat2 = data_train.copy()
        train_result_strat3 = data_train.copy()
        train_result_strat4 = data_train.copy()
        ## calculate scores
        train_result_strat1["SCORE"] = train_result_strat1.apply(lambda x: PIAScore.score(x, positives, negatives, "+"), axis = 1)
        train_result_strat2["SCORE"] = train_result_strat2.apply(lambda x: PIAScore.score(x, positives, negatives, "++"), axis = 1)
        train_result_strat3["SCORE"] = train_result_strat3.apply(lambda x: PIAScore.score(x, positives, negatives, "+-"), axis = 1)
        train_result_strat4["SCORE"] = train_result_strat4.apply(lambda x: PIAScore.score(x, positives, negatives, "++--"), axis = 1)
        ## sort data
        train_result_strat1_sorted = train_result_strat1.sort_values(by = "SCORE", ascending = False)
        train_result_strat2_sorted = train_result_strat2.sort_values(by = "SCORE", ascending = False)
        train_result_strat3_sorted = train_result_strat3.sort_values(by = "SCORE", ascending = False)
        train_result_strat4_sorted = train_result_strat4.sort_values(by = "SCORE", ascending = False)
        # determine cutoffs
        cutoff_strat1 = get_cutoff(train_result_strat1["LABEL"].to_list(), train_result_strat1["SCORE"].to_list())[0]
        cutoff_strat2 = get_cutoff(train_result_strat2["LABEL"].to_list(), train_result_strat2["SCORE"].to_list())[0]
        cutoff_strat3 = get_cutoff(train_result_strat3["LABEL"].to_list(), train_result_strat3["SCORE"].to_list())[0]
        cutoff_strat4 = get_cutoff(train_result_strat4["LABEL"].to_list(), train_result_strat4["SCORE"].to_list())[0]
        ## make data copies
        val_result_strat1 = data_val.copy()
        val_result_strat2 = data_val.copy()
        val_result_strat3 = data_val.copy()
        val_result_strat4 = data_val.copy()
        ## calculate scores
        val_result_strat1["SCORE"] = val_result_strat1.apply(lambda x: PIAScore.score(x, positives, negatives, "+"), axis = 1)
        val_result_strat2["SCORE"] = val_result_strat2.apply(lambda x: PIAScore.score(x, positives, negatives, "++"), axis = 1)
        val_result_strat3["SCORE"] = val_result_strat3.apply(lambda x: PIAScore.score(x, positives, negatives, "+-"), axis = 1)
        val_result_strat4["SCORE"] = val_result_strat4.apply(lambda x: PIAScore.score(x, positives, negatives, "++--"), axis = 1)
        ## sort data
        val_result_strat1_sorted = val_result_strat1.sort_values(by = "SCORE", ascending = False)
        val_result_strat2_sorted = val_result_strat2.sort_values(by = "SCORE", ascending = False)
        val_result_strat3_sorted = val_result_strat3.sort_values(by = "SCORE", ascending = False)
        val_result_strat4_sorted = val_result_strat4.sort_values(by = "SCORE", ascending = False)
        ## make data copies
        test_result_strat1 = data_test.copy()
        test_result_strat2 = data_test.copy()
        test_result_strat3 = data_test.copy()
        test_result_strat4 = data_test.copy()
        ## calculate scores
        test_result_strat1["SCORE"] = test_result_strat1.apply(lambda x: PIAScore.score(x, positives, negatives, "+"), axis = 1)
        test_result_strat2["SCORE"] = test_result_strat2.apply(lambda x: PIAScore.score(x, positives, negatives, "++"), axis = 1)
        test_result_strat3["SCORE"] = test_result_strat3.apply(lambda x: PIAScore.score(x, positives, negatives, "+-"), axis = 1)
        test_result_strat4["SCORE"] = test_result_strat4.apply(lambda x: PIAScore.score(x, positives, negatives, "++--"), axis = 1)
        ## sort data
        test_result_strat1_sorted = test_result_strat1.sort_values(by = "SCORE", ascending = False)
        test_result_strat2_sorted = test_result_strat2.sort_values(by = "SCORE", ascending = False)
        test_result_strat3_sorted = test_result_strat3.sort_values(by = "SCORE", ascending = False)
        test_result_strat4_sorted = test_result_strat4.sort_values(by = "SCORE", ascending = False)
        # evaluation
        self.train_results = {"TRAIN":
                                {
                                "+": PIAScore.get_metrics(train_result_strat1, cutoff_strat1),
                                "++": PIAScore.get_metrics(train_result_strat2, cutoff_strat2),
                                "+-": PIAScore.get_metrics(train_result_strat3, cutoff_strat3),
                                "++--": PIAScore.get_metrics(train_result_strat4, cutoff_strat4)
                                },
                              "VAL":
                                {
                                "+": PIAScore.get_metrics(val_result_strat1, cutoff_strat1),
                                "++": PIAScore.get_metrics(val_result_strat2, cutoff_strat2),
                                "+-": PIAScore.get_metrics(val_result_strat3, cutoff_strat3),
                                "++--": PIAScore.get_metrics(val_result_strat4, cutoff_strat4)
                                },
                              "TEST":
                              {
                              "+": PIAScore.get_metrics(test_result_strat1, cutoff_strat1),
                              "++": PIAScore.get_metrics(test_result_strat2, cutoff_strat2),
                              "+-": PIAScore.get_metrics(test_result_strat3, cutoff_strat3),
                              "++--": PIAScore.get_metrics(test_result_strat4, cutoff_strat4)
                              }
                            }

        # set optimal cutoff and strategy
        if strat == "strat1":
            self.strategy = "+"
            self.cutoff = cutoff_strat1
        elif strat == "strat2":
            self.strategy = "++"
            self.cutoff = cutoff_strat2
        elif strat == "strat3":
            self.strategy = "+-"
            self.cutoff = cutoff_strat3
        elif strat == "strat4":
            self.strategy = "++--"
            self.cutoff = cutoff_strat4
        # this should never happen
        else:
            warnings.warn("Strategy key not detected! This should not happen!", UserWarning)

        # cleanup
        shutil.rmtree("piamodel_structures_tmp")
        if not keep_files:
            if sdf_file_2 is not None:
                os.remove("piamodel_combo_tmp.sdf")
            os.remove(pdb_base_structure + "_cleaned.pdb")
            os.remove("piamodel_data_train_tmp.csv")
            os.remove("piamodel_data_val_tmp.csv")
            os.remove("piamodel_data_test_tmp.csv")
            os.remove("piamodel_features_tmp.csv")

        # return results
        return self.train_results

    # print a summary of metrics
    def summary(self):

        """
        -- DESCRIPTION --
        """

        if self.train_results is not None:
            print("----- TRAINING DATA SUMMARY -----")
            print("STRATEGY +:")
            print("    ACC: ", self.train_results["TRAIN"]["+"]["ACC"])
            print("    FPR: ", self.train_results["TRAIN"]["+"]["FPR"])
            print("    AUC: ", self.train_results["TRAIN"]["+"]["AUC"])
            print("    Ya: ", self.train_results["TRAIN"]["+"]["Ya"])
            print("    EF: ", self.train_results["TRAIN"]["+"]["EF"])
            print("    REF: ", self.train_results["TRAIN"]["+"]["REF"])
            print("STRATEGY ++:")
            print("    ACC: ", self.train_results["TRAIN"]["++"]["ACC"])
            print("    FPR: ", self.train_results["TRAIN"]["++"]["FPR"])
            print("    AUC: ", self.train_results["TRAIN"]["++"]["AUC"])
            print("    Ya: ", self.train_results["TRAIN"]["++"]["Ya"])
            print("    EF: ", self.train_results["TRAIN"]["++"]["EF"])
            print("    REF: ", self.train_results["TRAIN"]["++"]["REF"])
            print("STRATEGY +-:")
            print("    ACC: ", self.train_results["TRAIN"]["+-"]["ACC"])
            print("    FPR: ", self.train_results["TRAIN"]["+-"]["FPR"])
            print("    AUC: ", self.train_results["TRAIN"]["+-"]["AUC"])
            print("    Ya: ", self.train_results["TRAIN"]["+-"]["Ya"])
            print("    EF: ", self.train_results["TRAIN"]["+-"]["EF"])
            print("    REF: ", self.train_results["TRAIN"]["+-"]["REF"])
            print("STRATEGY ++--:")
            print("    ACC: ", self.train_results["TRAIN"]["++--"]["ACC"])
            print("    FPR: ", self.train_results["TRAIN"]["++--"]["FPR"])
            print("    AUC: ", self.train_results["TRAIN"]["++--"]["AUC"])
            print("    Ya: ", self.train_results["TRAIN"]["++--"]["Ya"])
            print("    EF: ", self.train_results["TRAIN"]["++--"]["EF"])
            print("    REF: ", self.train_results["TRAIN"]["++--"]["REF"])
            print("----- VALIDATION DATA SUMMARY -----")
            print("STRATEGY +:")
            print("    ACC: ", self.train_results["VAL"]["+"]["ACC"])
            print("    FPR: ", self.train_results["VAL"]["+"]["FPR"])
            print("    AUC: ", self.train_results["VAL"]["+"]["AUC"])
            print("    Ya: ", self.train_results["VAL"]["+"]["Ya"])
            print("    EF: ", self.train_results["VAL"]["+"]["EF"])
            print("    REF: ", self.train_results["VAL"]["+"]["REF"])
            print("STRATEGY ++:")
            print("    ACC: ", self.train_results["VAL"]["++"]["ACC"])
            print("    FPR: ", self.train_results["VAL"]["++"]["FPR"])
            print("    AUC: ", self.train_results["VAL"]["++"]["AUC"])
            print("    Ya: ", self.train_results["VAL"]["++"]["Ya"])
            print("    EF: ", self.train_results["VAL"]["++"]["EF"])
            print("    REF: ", self.train_results["VAL"]["++"]["REF"])
            print("STRATEGY +-:")
            print("    ACC: ", self.train_results["VAL"]["+-"]["ACC"])
            print("    FPR: ", self.train_results["VAL"]["+-"]["FPR"])
            print("    AUC: ", self.train_results["VAL"]["+-"]["AUC"])
            print("    Ya: ", self.train_results["VAL"]["+-"]["Ya"])
            print("    EF: ", self.train_results["VAL"]["+-"]["EF"])
            print("    REF: ", self.train_results["VAL"]["+-"]["REF"])
            print("STRATEGY ++--:")
            print("    ACC: ", self.train_results["VAL"]["++--"]["ACC"])
            print("    FPR: ", self.train_results["VAL"]["++--"]["FPR"])
            print("    AUC: ", self.train_results["VAL"]["++--"]["AUC"])
            print("    Ya: ", self.train_results["VAL"]["++--"]["Ya"])
            print("    EF: ", self.train_results["VAL"]["++--"]["EF"])
            print("    REF: ", self.train_results["VAL"]["++--"]["REF"])
            print("----- TEST DATA SUMMARY -----")
            print("STRATEGY +:")
            print("    ACC: ", self.train_results["TEST"]["+"]["ACC"])
            print("    FPR: ", self.train_results["TEST"]["+"]["FPR"])
            print("    AUC: ", self.train_results["TEST"]["+"]["AUC"])
            print("    Ya: ", self.train_results["TEST"]["+"]["Ya"])
            print("    EF: ", self.train_results["TEST"]["+"]["EF"])
            print("    REF: ", self.train_results["TEST"]["+"]["REF"])
            print("STRATEGY ++:")
            print("    ACC: ", self.train_results["TEST"]["++"]["ACC"])
            print("    FPR: ", self.train_results["TEST"]["++"]["FPR"])
            print("    AUC: ", self.train_results["TEST"]["++"]["AUC"])
            print("    Ya: ", self.train_results["TEST"]["++"]["Ya"])
            print("    EF: ", self.train_results["TEST"]["++"]["EF"])
            print("    REF: ", self.train_results["TEST"]["++"]["REF"])
            print("STRATEGY +-:")
            print("    ACC: ", self.train_results["TEST"]["+-"]["ACC"])
            print("    FPR: ", self.train_results["TEST"]["+-"]["FPR"])
            print("    AUC: ", self.train_results["TEST"]["+-"]["AUC"])
            print("    Ya: ", self.train_results["TEST"]["+-"]["Ya"])
            print("    EF: ", self.train_results["TEST"]["+-"]["EF"])
            print("    REF: ", self.train_results["TEST"]["+-"]["REF"])
            print("STRATEGY ++--:")
            print("    ACC: ", self.train_results["TEST"]["++--"]["ACC"])
            print("    FPR: ", self.train_results["TEST"]["++--"]["FPR"])
            print("    AUC: ", self.train_results["TEST"]["++--"]["AUC"])
            print("    Ya: ", self.train_results["TEST"]["++--"]["Ya"])
            print("    EF: ", self.train_results["TEST"]["++--"]["EF"])
            print("    REF: ", self.train_results["TEST"]["++--"]["REF"])
        else:
            print("No model statistics are available!")

        return

    # save model to file
    def save(self,
             filename):

        """
        -- DESCRIPTION --
        """

        # define model as dict
        model = {"positives": self.positives,
                 "negatives": self.negatives,
                 "strategy": self.strategy,
                 "cutoff": self.cutoff,
                 "results": self.train_results}

        # save in json format
        piam_file = filename + ".piam"
        with open(piam_file, "w", encoding = "utf-8") as f:
            json.dump(model, f)
            f.close()

        # return filename
        return piam_file

    # predict a single pdb file
    def predict_pdb(self,
                    pdb_file):

        """
        -- DESCRIPTION --
        """

        # characterize complex
        mol = PDBComplex()
        mol.load_pdb(pdb_file)
        for ligand in mol.ligands:
            mol.characterize_complex(ligand)

        # initialize interactions
        Salt_Bridges = []
        Hydrogen_Bonds = []
        Pi_Stacking = []
        Pi_Cation_Interactions = []
        Hydrophobic_Contacts = []
        Halogen_Bonds = []
        Water_Bridges = []
        Metal_Complexes = []

        # iterate over interaction sets
        for key in mol.interaction_sets:
            iHet_ID, iChain, iPosition = key.split(":")
            # discard suspicious ligands
            if iHet_ID.strip().upper() in exclusion_list:
                continue
            # discard uninteressted chains
            # if iChain != "A":
            #    continue
            interaction = mol.interaction_sets[key]

            # get interaction residues
            # SALT BRIDGES
            tmp_salt_bridges = interaction.saltbridge_lneg + interaction.saltbridge_pneg
            Salt_Bridges = Salt_Bridges + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in tmp_salt_bridges if i.restype not in exclude]
            # HYDROGEN BONDS
            tmp_h_bonds = interaction.hbonds_pdon + interaction.hbonds_ldon
            Hydrogen_Bonds = Hydrogen_Bonds + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in tmp_h_bonds if i.restype not in exclude]
            # PI STACKING
            Pi_Stacking = Pi_Stacking + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in interaction.pistacking if i.restype not in exclude]
            # PI CATION INTERACTION
            tmp_pication = interaction.pication_laro + interaction.pication_paro
            Pi_Cation_Interactions = Pi_Cation_Interactions + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in tmp_pication if i.restype not in exclude]
            # HYDROPHOBIC CONTACTS
            Hydrophobic_Contacts = Hydrophobic_Contacts + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in interaction.hydrophobic_contacts if i.restype not in exclude]
            # HALOGEN BONDS
            Halogen_Bonds = Halogen_Bonds + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in interaction.halogen_bonds if i.restype not in exclude]
            # WATER BRIDGES
            Water_Bridges = Water_Bridges + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in interaction.water_bridges if i.restype not in exclude]
            # METAL COMPLEXES
            Metal_Complexes = Metal_Complexes + [''.join([str(i.restype), str(i.resnr), str(i.reschain)]) for i in interaction.metal_complexes if i.restype not in exclude]

        # pool interactions
        interactions1 = ["Salt_Bridge:" + i for i in Salt_Bridges] + ["Hydrogen_Bond:" + i for i in Hydrogen_Bonds]
        interactions2 = ["Pi-Stacking:" + i for i in Pi_Stacking] + ["Pi-Cation_Interaction:" + i for i in Pi_Cation_Interactions]
        interactions3 = ["Hydrophobic_Interaction:" + i for i in set(Hydrophobic_Contacts)] + ["Halogen_Bond:" + i for i in Halogen_Bonds]
        interactions4 = ["Water_Bridge:" + i for i in Water_Bridges] + ["Metal_Complexation:" + i for i in Metal_Complexes]
        all_interactions = interactions1 + interactions2 + interactions3 + interactions4

        # score
        score = 0
        # strategy 1
        if self.strategy == "+":
            for interaction in positives:
                if interaction in all_interactions:
                    score = score + 1
        # strategy 2
        elif self.strategy == "++":
            for interaction in positives:
                score = score + all_interactions.count(interaction)
        # strategy 3
        elif self.strategy == "+-":
            for interaction in positives:
                if interaction in all_interactions:
                    score = score + 1
            for interaction in negatives:
                if if interaction in all_interactions:
                    score = score - 1
        # strategy 4
        elif self.strategy == "++--":
            for interaction in positives:
                score = score + all_interactions.count(interaction)
            for interaction in negatives:
                score = score - all_interactions.count(interaction)
        else:
            pass

        # predict
        if score >= self.cutoff:
            prediction = "active"
        else:
            prediction = "inactive"

        # return filename, score and prediction
        return {"name": pdb_file, "score": score, "prediction": prediction}

    # predict multiple ligands from a sdf file
    def predict_sdf(self,
                    pdb_base_structure,
                    sdf_file):

        """
        -- DESCRIPTION --
        """

        # init return values
        names = []
        scores = []
        predictions = []

        # create necessary directories
        structures_directory = "piamodel_structures_tmp"
        structures_path = os.path.join(os.getcwd(), structures_directory)
        os.mkdir(structures_path)

        # prepare files
        p = Preparation()
        # clean PDB file
        pdb = p.remove_ligands(pdb_base_structure, pdb_base_structure + "_cleaned.pdb")
        # read ligands from docking SDF file
        ligands = p.get_ligands(sdf_file)
        # get ligand names
        sdf_metainfo = p.get_sdf_metainfo(sdf_file)
        ligand_names = sdf_metainfo["names"]
        # write ligands into PDB files
        structures = p.add_ligands_multi(pdb_base_structure + "_cleaned.pdb", "piamodel_structures_tmp", ligands)
        # analyze structures
        for i, structure in enumerate(structures):
            result = self.predict_pdb(structure)
            names.append(ligand_names[i])
            scores.append(result["score"])
            prediction.append(result["prediction"])

        # cleanup
        shutil.rmtree("piamodel_structures_tmp")
        os.remove(pdb_base_structure + "_cleaned.pdb")

        # return ligand names, scores and predictions
        return {"names": names, "scores": scores, "predictions": predictions}
