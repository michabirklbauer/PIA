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
from PIA import PIAScore
from PIA.PIA import PIA
from PIA.PIA import Preparation
from PIA.PIA import Scoring
from PIA.PIA import Comparison
from PIA.PIA import exclusion_list
from plip.structure.preparation import PDBComplex

class NoLabelsError(RuntimeError):
    """
    -- DESCRIPTION --
    Raised if labelling condition is not recognized.
    """
    pass

class PIAModel:
    """
    -- DESCRIPTION --
    Model class that facilitates creating, training, changing, saving and
    loading of a scoring model.
    """

    # mandatory attributes, either set via constructor or trained
    positives = None
    negatives = None
    strategy = None
    cutoff = None
    # model statistics (optional, available if loaded from file or trained)
    statistics = None

    # trainable attributes (optional)
    train_results = None
    plot_train = None
    plot_val = None
    plot_test = None

    # constructor - initialize a scoring model
    def __init__(self,
                 positives = None,
                 negatives = None,
                 strategy = None,
                 cutoff = None,
                 filename = None):

        """
        -- DESCRIPTION --
        Constructor for creating a scoring model. A model can either be created
        by:
          (i)   specifying the PARAMS "positives", "negatives", "strategy" and
                "cutoff", if strategy is "+" or "++" -> "negatives" doesn't need
                to be specified
          (ii)  specifying the PARAM "filename" in which case the model will be
                loaded from file
          (iii) calling an empty constructor and run PIAModel.train() method
                afterwards
        Depending on the choice of model creation (initializing, loading,
        training) not all ATTRIBUTES may be available from the get-go.
          PARAMS:
            - positives (list of strings):
                list of interactions with positive impact on the score,
                interactions have to be in the format of Hydrogen_Bond:ASP351A
                for example. Therefore it's "interactions type: residue position
                chain" without spaces and residue and chain in upper case. This
                PARAM has to be specified if the model is not loaded from file
                or trained.
            - negatives (list of strings):
                list of interactions with negative impact on the score, in the
                same format as for PARAM "positives". "negatives" has to be
                specified if PARAM "strategy" is not "+" or "++", not loaded
                from file or trained.
            - strategy (string): one of "+", "++", "+-", "++--".
                scoring strategy to be applied, has to be specifed unless model
                is loaded from file or trained
            - cutoff (int):
                cutoff applied to predict if a complex is active (score equal or
                above cutoff) or inactive (below cutoff), has to be specified
                unless model is loaded from file or trained
            - filename (string):
                a valid path/filename to PIAModel file, only has to be specified
                if the model should be loaded from the file
          RETURNS:
            None
          ATTRIBUTES:
            - positives (list of strings):
                see PARAMS, available immediately in (i), (ii) or after training
                (iii)
            - negatives (list of strings):
                see PARAMS, available immediately in (i), (ii) or after training
                (iii)
            - strategy (string):
                see PARAMS, available immediately in (i), (ii) or after training
                (iii)
            - cutoff (int):
                see PARAMS, available immediately in (i), (ii) or after training
                (iii)
            - statistics (dict of dicts):
                quality metrics and parameters from training that are available
                immediately if model was loaded from file (ii) or after training
                (iii)
                  -> see line 387 for complete structure of the dictionary
            - train_results (dict of dicts):
                additional quality metrics from training that are only available
                if the model was trained (iii)
                  -> see line 365 for complete structure of the dictionary
            - plot_train (matplotlib.pyplot.figure object):
                plot of interaction frequencies of actives and inactives in the
                training partition, only available if model was trained (iii)
            - plot_val (matplotlib.pyplot.figure object):
                plot of interaction frequencies of actives and inactives in the
                validation partition, only available if model was trained (iii)
            - plot_test (matplotlib.pyplot.figure object):
                plot of interaction frequencies of actives and inactives in the
                test partition, only available if model was trained (iii)
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
            self.statistics = data["statistics"]
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
              poses = "best",
              test_size = 0.3,
              val_size = 0.3,
              labels_by = "name",
              condition_operator = ">=",
              condition_value = 1000,
              plot_prefix = None,
              keep_files = False,
              block = False,
              verbose = 1):

        """
        -- DESCRIPTION --
        Train a model from one (or two) SDF files and a PDB base structure.
          PARAMS:
            - pdb_base_structure (string):
                path/filename of the host PDB file that was used for docking of
                the ligands. Can be complexed with a ligand since it will be
                cleaned before processing anyway
            - sdf_file_1 (string):
                path/filename of the SDF file containing ligands coordiantes,
                ligand names should contain "inactive" or "decoy" in their name
                if to classify them as inactive, active ligands MUST NOT contain
                these terms
            - sdf_file_2 (string):
                optional path/filename to a second SDF file containing ligands
                coordiantes, ligand names should contain "inactive" or "decoy"
                in their name if to classify them as inactive, active ligands
                MUST NOT contain these terms
                DEFAULT: None
            - poses (string): one of "all", "best".
                if multiple poses of the same ligand are present (e.g. from
                docking) either all are analyzed or just the best (the pose with
                the most interactions).
                "best" only works if the molecules are named in the GOLD schema
                e.g. 'LIG1_active|ligand|sdf|1|dock1'
                DEFAULT: "best"
            - test_size (float; interval (0,1)):
                size of the test partition / fraction of the whole dataset
                DEFAULT: 0.3 (30% of the whole dataset used for testing)
            - val_size (float; intervall (0,1)):
                size of the validation partition fraction of the remaining 70%
                that is not used for testing
                DEFAULT: 0.3 (30% of the remaining 70% is used for validation)
            - labels_by (string): one of "name", "ic50"
                label molecules as active/inactive based on name or ic50 value,
                if "ic50" is specified the user may supply a
                "condition_operator" and "condition_value" (see below)
                DEFAULT: "name" (label by name)
            - condition_operator (string): one of "==", "!=", "<=", "<", ">=",
                                                  ">".
                The molecule is labelled as inactive if the IC50 value is
                "condition_operator" "condition_value" e.g. if
                "condition_operator" is ">=" and "condition_value" is "1000"
                then all molecules where "IC50 >= 1000" are labelled as inactive
                with subfix "_decoy". Any molecule that does not have an IC50
                value is not labelled
                DEFAULT: ">="
            - condition_value (int or float): reference value
                The molecule is labelled as inactive if the IC50 value is
                "condition_operator" "condition_value" e.g. if
                "condition_operator" is ">=" and "condition_value" is "1000"
                then all molecules where "IC50 >= 1000" are labelled as inactive
                with subfix "_decoy". Any molecule that does not have an IC50
                value is not labelled
                DEFAULT: 1000
            - plot_prefix (string):
                optional path/filename prefix for plots if they should be saved,
                if None is supplied the plots will not be saved
                DEFAULT: None
            - keep_files (bool):
                if temporary files that are created during training should be
                kept after the training is finished or not
                DEFAULT: False (files are not kept)
            - block (bool):
                if matplotlib should wait for closing all plot windows before
                continuing or return immediately
                DEFAULT: False (matplotlib returns immediately and plots will be
                         be generated when the training process is finished)
            - verbose (bool/0 or 1):
                print additional training infos to std ouput
                DEFAULT: 1 (information will be printed to std output)
          RETURNS:
            - train_results (dict of dicts):
                quality metrics from training
                  -> see line 365 for complete structure of the dictionary
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
        if labels_by == "name":
            ligands = p.get_ligands(filename)
            sdf_metainfo = p.get_sdf_metainfo(filename)
            ligand_names = sdf_metainfo["names"]
        elif labels_by == "ic50":
            ligands_info = p.get_labeled_names(filename, condition_operator, condition_value)
            ligands = ligands_info["ligands"]
            ligand_names = ligands_info["names"]
        else:
            raise NoLabelsError(str(labels_by) + " not recognized! Can't label molecules!")
        structures = p.add_ligands_multi(pdb_base_structure + "_cleaned.pdb", "piamodel_structures_tmp", ligands, verbose = verbose)
        # PIA
        result = PIA(structures, ligand_names = ligand_names, poses = poses, path = "current", verbose = verbose)
        # scoring preparation
        s = Scoring(result.pdb_entry_results, result.best_pdb_entries)
        # generate datasets
        data_train, data_val, data_test = s.generate_datasets(test_size = test_size,
                                                              val_size = val_size,
                                                              train_output = "piamodel_data_train_tmp.csv",
                                                              val_output = "piamodel_data_val_tmp.csv",
                                                              test_output = "piamodel_data_test_tmp.csv")
        # compare actives and inactives
        ia_info = s.get_actives_inactives()
        comp_train = s.compare(partition = "train")
        comp_val = s.compare(partition = "val")
        comp_test = s.compare(partition = "test")
        # get feature names and information
        features = s.get_feature_information(filename = "piamodel_features_tmp.csv")
        # plot comparisons, if plot_prefix is specified save them
        if plot_prefix is not None:
            self.plot_train = comp_train.plot("Actives vs. Inactives - Training Set", filename = plot_prefix + "_comparison_train.png", block = block)
            self.plot_val = comp_val.plot("Actives vs. Inactives - Validation Set", filename = plot_prefix + "_comparison_val.png", block = block)
            self.plot_test = comp_test.plot("Actives vs. Inactives - Test Set", filename = plot_prefix + "_comparison_test.png", block = block)
        else:
            self.plot_train = comp_train.plot("Actives vs. Inactives - Training Set", block = block)
            self.plot_val = comp_val.plot("Actives vs. Inactives - Validation Set", block = block)
            self.plot_test = comp_test.plot("Actives vs. Inactives - Test Set", block = block)

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
        positives = self.positives
        negatives = self.negatives
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
        cutoff_strat1 = PIAScore.get_cutoff(train_result_strat1["LABEL"].to_list(), train_result_strat1["SCORE"].to_list())[0]
        cutoff_strat2 = PIAScore.get_cutoff(train_result_strat2["LABEL"].to_list(), train_result_strat2["SCORE"].to_list())[0]
        cutoff_strat3 = PIAScore.get_cutoff(train_result_strat3["LABEL"].to_list(), train_result_strat3["SCORE"].to_list())[0]
        cutoff_strat4 = PIAScore.get_cutoff(train_result_strat4["LABEL"].to_list(), train_result_strat4["SCORE"].to_list())[0]
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
        # set optimal cutoff and strategy
        if strat == "strat1":
            best_strategy = "+"
            self.strategy = "+"
            self.cutoff = cutoff_strat1
        elif strat == "strat2":
            best_strategy = "++"
            self.strategy = "++"
            self.cutoff = cutoff_strat2
        elif strat == "strat3":
            best_strategy = "+-"
            self.strategy = "+-"
            self.cutoff = cutoff_strat3
        elif strat == "strat4":
            best_strategy = "+--"
            self.strategy = "++--"
            self.cutoff = cutoff_strat4
        # this should never happen
        else:
            warnings.warn("Strategy key not detected! This should not happen!", UserWarning)
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
        self.statistics = {"STRAT":
                            {
                            "best_strategy": best_strategy,
                            "cutoffs":
                                {
                                "+": cutoff_strat1,
                                "++": cutoff_strat2,
                                "+-": cutoff_strat3,
                                "++--": cutoff_strat4
                                }
                            },
                           "TRAIN":
                            {
                            "+": PIAScore.get_metrics(train_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(train_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(train_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(train_result_strat4, cutoff_strat4, pretty_print = True)
                            },
                           "VAL":
                            {
                            "+": PIAScore.get_metrics(val_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(val_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(val_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(val_result_strat4, cutoff_strat4, pretty_print = True)
                            },
                           "TEST":
                            {
                            "+": PIAScore.get_metrics(test_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(test_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(test_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(test_result_strat4, cutoff_strat4, pretty_print = True)
                            }
                           }

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

    # train model from csv files (optional)
    def train_from_csv(self,
                       data_train_csv,
                       data_val_csv,
                       data_test_csv,
                       features_csv,
                       verbose = 1):

        """
        -- DESCRIPTION --
        Train a scoring model from preprocessed CSV files.
          PARAMS:
            - data_test_csv (string):
                name of the csv file for the training partition
            - data_val_csv (string):
                name of the csv file for the validation partition
            - data_test_csv (string):
                name of the csv file for the test partition
            - features_csv (string):
                name of the csv file for feature information
            - verbose (bool/0 or 1):
                print additional training infos to std ouput
                DEFAULT: 1 (information will be printed to std output)
          RETURNS:
            - train_results (dict of dicts):
                quality metrics from training
                  -> see line 365 for complete structure of the dictionary
        """

        # load data
        data_train = pd.read_csv(data_train_csv)
        data_val = pd.read_csv(data_val_csv)
        data_test = pd.read_csv(data_test_csv)
        features = pd.read_csv(features_csv)

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
        positives = self.positives
        negatives = self.negatives
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
        cutoff_strat1 = PIAScore.get_cutoff(train_result_strat1["LABEL"].to_list(), train_result_strat1["SCORE"].to_list())[0]
        cutoff_strat2 = PIAScore.get_cutoff(train_result_strat2["LABEL"].to_list(), train_result_strat2["SCORE"].to_list())[0]
        cutoff_strat3 = PIAScore.get_cutoff(train_result_strat3["LABEL"].to_list(), train_result_strat3["SCORE"].to_list())[0]
        cutoff_strat4 = PIAScore.get_cutoff(train_result_strat4["LABEL"].to_list(), train_result_strat4["SCORE"].to_list())[0]
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
        # set optimal cutoff and strategy
        if strat == "strat1":
            best_strategy = "+"
            self.strategy = "+"
            self.cutoff = cutoff_strat1
        elif strat == "strat2":
            best_strategy = "++"
            self.strategy = "++"
            self.cutoff = cutoff_strat2
        elif strat == "strat3":
            best_strategy = "+-"
            self.strategy = "+-"
            self.cutoff = cutoff_strat3
        elif strat == "strat4":
            best_strategy = "+--"
            self.strategy = "++--"
            self.cutoff = cutoff_strat4
        # this should never happen
        else:
            warnings.warn("Strategy key not detected! This should not happen!", UserWarning)
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
        self.statistics = {"STRAT":
                            {
                            "best_strategy": best_strategy,
                            "cutoffs":
                                {
                                "+": cutoff_strat1,
                                "++": cutoff_strat2,
                                "+-": cutoff_strat3,
                                "++--": cutoff_strat4
                                }
                            },
                           "TRAIN":
                            {
                            "+": PIAScore.get_metrics(train_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(train_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(train_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(train_result_strat4, cutoff_strat4, pretty_print = True)
                            },
                           "VAL":
                            {
                            "+": PIAScore.get_metrics(val_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(val_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(val_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(val_result_strat4, cutoff_strat4, pretty_print = True)
                            },
                           "TEST":
                            {
                            "+": PIAScore.get_metrics(test_result_strat1, cutoff_strat1, pretty_print = True),
                            "++": PIAScore.get_metrics(test_result_strat2, cutoff_strat2, pretty_print = True),
                            "+-": PIAScore.get_metrics(test_result_strat3, cutoff_strat3, pretty_print = True),
                            "++--": PIAScore.get_metrics(test_result_strat4, cutoff_strat4, pretty_print = True)
                            }
                           }

        # return results
        return self.train_results

    # print a summary of metrics
    def summary(self,
                filename = None):

        """
        -- DESCRIPTION --
        Print a summary of parameters and quality metrics of the model. Only
        available if model was loaded from file (ii) or trained (iii).
          PARAMS:
            - filename (string): filename if/where generated summary should be
                                 saved. If no filename is given the summary is
                                 not saved
          RETURNS:
            None
        """

        if self.statistics is not None:
            s = ""
            s = s + "#############################################\n"
            s = s + "#############################################\n"
            s = s + "---------------- STRATEGIES -----------------\n"
            s = s + "Best Strategy: " + str(self.statistics["STRAT"]["best_strategy"]) + "\n"
            s = s + "Cutoff Strategy +: " + str(self.statistics["STRAT"]["cutoffs"]["+"]) + "\n"
            s = s + "Cutoff Strategy ++: " + str(self.statistics["STRAT"]["cutoffs"]["++"]) + "\n"
            s = s + "Cutoff Strategy +-: " + str(self.statistics["STRAT"]["cutoffs"]["+-"]) + "\n"
            s = s + "Cutoff Strategy ++--: " + str(self.statistics["STRAT"]["cutoffs"]["++--"]) + "\n"
            s = s + "#############################################\n"
            s = s + "----------- TRAINING DATA SUMMARY -----------\n"
            s = s + "#############################################\n"
            s = s + "STRATEGY +:\n"
            s = s + "    ACC: " + str(self.statistics["TRAIN"]["+"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TRAIN"]["+"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TRAIN"]["+"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TRAIN"]["+"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TRAIN"]["+"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TRAIN"]["+"]["REF"]) + "\n"
            s = s + "STRATEGY ++:\n"
            s = s + "    ACC: " + str(self.statistics["TRAIN"]["++"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TRAIN"]["++"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TRAIN"]["++"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TRAIN"]["++"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TRAIN"]["++"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TRAIN"]["++"]["REF"]) + "\n"
            s = s + "STRATEGY +-:\n"
            s = s + "    ACC: " + str(self.statistics["TRAIN"]["+-"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TRAIN"]["+-"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TRAIN"]["+-"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TRAIN"]["+-"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TRAIN"]["+-"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TRAIN"]["+-"]["REF"]) + "\n"
            s = s + "STRATEGY ++--:\n"
            s = s + "    ACC: " + str(self.statistics["TRAIN"]["++--"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TRAIN"]["++--"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TRAIN"]["++--"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TRAIN"]["++--"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TRAIN"]["++--"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TRAIN"]["++--"]["REF"]) + "\n"
            s = s + "#############################################\n"
            s = s + "---------- VALIDATION DATA SUMMARY ----------\n"
            s = s + "#############################################\n"
            s = s + "STRATEGY +:\n"
            s = s + "    ACC: " + str(self.statistics["VAL"]["+"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["VAL"]["+"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["VAL"]["+"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["VAL"]["+"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["VAL"]["+"]["EF"]) + "\n"
            s = s + "    REF: "+ str(self.statistics["VAL"]["+"]["REF"]) + "\n"
            s = s + "STRATEGY ++:\n"
            s = s + "    ACC: " + str(self.statistics["VAL"]["++"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["VAL"]["++"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["VAL"]["++"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["VAL"]["++"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["VAL"]["++"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["VAL"]["++"]["REF"]) + "\n"
            s = s + "STRATEGY +-:\n"
            s = s + "    ACC: " + str(self.statistics["VAL"]["+-"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["VAL"]["+-"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["VAL"]["+-"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["VAL"]["+-"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["VAL"]["+-"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["VAL"]["+-"]["REF"]) + "\n"
            s = s + "STRATEGY ++--:\n"
            s = s + "    ACC: " + str(self.statistics["VAL"]["++--"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["VAL"]["++--"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["VAL"]["++--"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["VAL"]["++--"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["VAL"]["++--"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["VAL"]["++--"]["REF"]) + "\n"
            s = s + "#############################################\n"
            s = s + "------------- TEST DATA SUMMARY -------------\n"
            s = s + "#############################################\n"
            s = s + "STRATEGY +:\n"
            s = s + "    ACC: " + str(self.statistics["TEST"]["+"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TEST"]["+"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TEST"]["+"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TEST"]["+"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TEST"]["+"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TEST"]["+"]["REF"]) + "\n"
            s = s + "STRATEGY ++:\n"
            s = s + "    ACC: " + str(self.statistics["TEST"]["++"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TEST"]["++"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TEST"]["++"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TEST"]["++"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TEST"]["++"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TEST"]["++"]["REF"]) + "\n"
            s = s + "STRATEGY +-:\n"
            s = s + "    ACC: " + str(self.statistics["TEST"]["+-"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TEST"]["+-"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TEST"]["+-"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TEST"]["+-"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TEST"]["+-"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TEST"]["+-"]["REF"]) + "\n"
            s = s + "STRATEGY ++--:\n"
            s = s + "    ACC: " + str(self.statistics["TEST"]["++--"]["ACC"]) + "\n"
            s = s + "    FPR: " + str(self.statistics["TEST"]["++--"]["FPR"]) + "\n"
            s = s + "    AUC: " + str(self.statistics["TEST"]["++--"]["AUC"]) + "\n"
            s = s + "    Ya: " + str(self.statistics["TEST"]["++--"]["Ya"]) + "\n"
            s = s + "    EF: " + str(self.statistics["TEST"]["++--"]["EF"]) + "\n"
            s = s + "    REF: " + str(self.statistics["TEST"]["++--"]["REF"]) + "\n"
            s = s + "#############################################\n"
            print(s)
            if filename is not None:
                with open(filename, "w", encoding = "utf-8") as f:
                    f.write(s)
                    f.close()
        else:
            print("No model statistics are available!")

        return

    # change scoring strategy
    def change_strategy(self,
                        strategy = "best"):

        """
        -- DESCRIPTION --
        Change the scoring strategy and cutoff. This method is only available
        if the model was loaded from file (ii) or trained (iii).
          PARAMS:
            - strategy (string): one of "best", "+", "++", "+-", "++--".
                scoring strategy to be applied, "best" refers to the best-on-
                validation strategy
                DEFAULT: "best"
          RETURNS:
            - list of changed strategy (string, index 0) and changed
              cutoff (int, index 1)
        """

        if self.statistics is not None:
            if strategy == "best":
                self.strategy = self.statistics["STRAT"]["best_strategy"]
                self.cutoff = int(self.statistics["STRAT"]["cutoffs"][self.strategy])
            elif strategy == "+":
                self.strategy = "+"
                self.cutoff = int(self.statistics["STRAT"]["cutoffs"][self.strategy])
            elif strategy == "++":
                self.strategy = "++"
                self.cutoff = int(self.statistics["STRAT"]["cutoffs"][self.strategy])
            elif strategy == "+-":
                self.strategy = "+-"
                self.cutoff = int(self.statistics["STRAT"]["cutoffs"][self.strategy])
            elif strategy == "++--":
                self.strategy = "++--"
                self.cutoff = int(self.statistics["STRAT"]["cutoffs"][self.strategy])
            else:
                pass

        return [self.strategy, self.cutoff]

    # save model to file
    def save(self,
             filename):

        """
        -- DESCRIPTION --
        Save model configuration to file.
          PARAMS:
            - filename (string): path/filename where the model should be saved
          RETURNS:
            - path/filename (string) of the created file
        """

        # define model as dict
        model = {"positives": self.positives,
                 "negatives": self.negatives,
                 "strategy": self.strategy,
                 "cutoff": self.cutoff,
                 "statistics": self.statistics}

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
        Score/Predict a single protein-ligand complex in PDB format.
          PARAMS:
            - pdb_file (string): path/filename of the complex in PDB format
          RETURNS:
            - dict with keys "name" (string, filename of the complex), "score"
              (int, the score of the complex), "prediction" (string, "active"
              or "inactive", the prediction of the complex) and "dataframe"
              containing the result as a pandas dataframe
        """

        # set PIA params
        exclude = ["LIG", "HOH"]

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
            for interaction in self.positives:
                if interaction in all_interactions:
                    score = score + 1
        # strategy 2
        elif self.strategy == "++":
            for interaction in self.positives:
                score = score + all_interactions.count(interaction)
        # strategy 3
        elif self.strategy == "+-":
            for interaction in self.positives:
                if interaction in all_interactions:
                    score = score + 1
            for interaction in self.negatives:
                if interaction in all_interactions:
                    score = score - 1
        # strategy 4
        elif self.strategy == "++--":
            for interaction in self.positives:
                score = score + all_interactions.count(interaction)
            for interaction in self.negatives:
                score = score - all_interactions.count(interaction)
        else:
            pass

        # predict
        if score >= self.cutoff:
            prediction = "active"
        else:
            prediction = "inactive"

        # create csv
        csv = pd.DataFrame({"NAME": [pdb_file], "SCORE": [score], "PREDICTION": [prediction]})

        # return filename, score and prediction
        return {"name": pdb_file, "score": score, "prediction": prediction, "dataframe": csv}

    # predict multiple ligands from a sdf file
    def predict_sdf(self,
                    pdb_base_structure,
                    sdf_file,
                    save_csv = False,
                    verbose = 1):

        """
        -- DESCRIPTION --
        Predict multiple protein-ligand complexes from a host PDB file and
        ligands in SDF format.
          PARAMS:
            - pdb_base_structure (string):
              path/filename of the host PDB file that was used for docking of
              the ligands. Can be complexed with a ligand since it will be
              cleaned before processing anyway
            - sdf_file (string):
              path/filename of the SDF file with ligand coordinates
            - save_csv (bool):
              if results should be saved as csv files, csv files are created in
              the current working directory with filename of the SDF file +
              ".csv"
              DEFAULT: False (results are not saved)
            - verbose (bool/0 or 1):
              should additional info be printed during the prediction process
              DEFAULT: 1 (information will be printed to std output)
          RETURNS:
            - dict of lists with keys "names" (list of strings, ligand names of
              the complexes), "scores" (list of ints, the scores of the
              complexes) and "predictions" (list of strings, "active" or
              "inactive", the predictions of the complexs) where
              dict["names"][i] corresponds to dict["scores"][i] and
              dict["predictions"][i], the results as a pandas dataframe via key
              "dataframe" (sorted by score, descending)
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
            predictions.append(result["prediction"])
            if verbose:
                print("Analyzed structure ", i + 1, "!")

        # cleanup
        shutil.rmtree("piamodel_structures_tmp")
        os.remove(pdb_base_structure + "_cleaned.pdb")

        # create csv
        csv = pd.DataFrame({"NAME": names, "SCORE": scores, "PREDICTION": predictions}).sort_values(by = "SCORE", ascending = False)

        # save csv
        if save_csv:
            csv.to_csv(sdf_file + ".csv")

        # return ligand names, scores, predictions and dataframe
        return {"names": names, "scores": scores, "predictions": predictions, "dataframe": csv}
