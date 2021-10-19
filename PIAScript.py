#!/usr/bin/env python3

# PIA - APPLICATION
# 2021 (c) Micha Johannes Birklbauer
# https://github.com/michabirklbauer/
# micha.birklbauer@gmail.com

version = "1.0.0"
date = "20211007"

"""
DESCRIPTION
PIAScript is a commandline application to apply PIA without the need of coding
your own workflows. PIAScript supports several input modes and can be tuned by
specifying the according arguments. See README.md for a complete description and
example usages!

ARGUMENTS:
-h --help: help
-m --mode: workflow, can be any of "extract", "compare", "score" and "predict"
-f --file: files to be processed
-c --cutoff: cutoff, optional for if interactions are provided for scoring
"""

import os
import math
import shutil
import argparse
import urllib.request as ur
from datetime import datetime
from PIA.PIAScore import *
from PIA.PIA import PIA as PIA
from PIA.PIA import Preparation as Preparation
from PIA.PIA import Comparison as Comparison
from PIA.PIAModel import PIAModel as PIAModel

#### -------------------------- HELPER FUNCTIONS -------------------------- ####

# get elements from a comma delimited txt file
def txt_to_list(txt_file):

    """
    -- DESCRIPTION --
    Return elements as list from a comma seperated txt file.
    """

    with open(txt_file, "r", encoding = "utf-8") as f:
        data = f.read()
        f.close()

    return [i.strip() for i in data.split(",")]

# get file types
def file_parser(list_of_files):

    """
    -- DESCRIPTION --
    Parse a list of filenames for the file extensions. Return needed file types
    as dictionary.
    """

    # file types processed by PIAScript
    pdb = None
    sdf1 = None
    sdf2 = None
    txt = None
    piam = None

    # extract file extensions
    for f in list_of_files:
        if f.split(".")[-1] == "pdb":
            if pdb == None:
                pdb = f
                continue
            else:
                continue
        elif f.split(".")[-1] == "sdf":
            if sdf1 == None:
                sdf1 = f
                continue
            elif sdf2 == None:
                sdf2 = f
                continue
            else:
                continue
        elif f.split(".")[-1] == "piam":
            if piam == None:
                piam = f
                continue
            else:
                continue
        else:
            if txt == None:
                txt = f
                continue
            else:
                continue

    return {"pdb": pdb, "sdf1": sdf1, "sdf2": sdf2, "txt": txt, "piam": piam}

#### -------------------------- PIASCRIPT MODES --------------------------- ####

# workflow extract - input mode 1: pdb codes
def extract_codes(list_of_codes):

    """
    -- DESCRIPTION --
    Download PDB entries and process them.
    """

    # create list of PDB links
    filenames = [i + ".pdb" if i.split(".")[-1] != "pdb" else i for i in list_of_codes]
    download_links = ["https://files.rcsb.org/download/" + i for i in filenames]

    # download files
    for i, link in enumerate(download_links):
        ur.urlretrieve(link, filenames[i])

    # extract interactions and frequencies
    output_name_prefix = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    result = PIA(filenames)
    p = result.plot("Analysis of PDB Codes", filename = output_name_prefix + "_analysis.png")
    r = result.save(output_name_prefix + "_analysis", True)
    c = result.to_csv(output_name_prefix + "_analysis.csv")

    # cleanup
    for f in filenames:
        os.remove(f)

    return result

# workflow extract - input mode 2: pdb files
def extract_pdbs(list_of_files):

    """
    -- DESCRIPTION --
    Process local PDB structures.
    """

    # extract interactions and frequencies
    output_name_prefix = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    result = PIA(list_of_files)
    p = result.plot("Analysis of PDB files", filename = output_name_prefix + "_analysis.png")
    r = result.save(output_name_prefix + "_analysis", True)
    c = result.to_csv(output_name_prefix + "_analysis.csv")

    return result

# workflow extract - input mode 3: sdf file
def extract_sdf(pdb_file, sdf_file):

    """
    -- DESCRIPTION --
    Process docked structures from a SDF file.
    """

    # create necessary directories
    structures_directory = "piascript_structures_tmp"
    structures_path = os.path.join(os.getcwd(), structures_directory)
    os.mkdir(structures_path)

    # extract interactions and frequencies
    output_name_prefix = sdf_file.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    p = Preparation()
    pdb = p.remove_ligands(pdb_file, pdb_file.split(".pdb")[0] + "_cleaned.pdb")
    ligands = p.get_ligands(sdf_file)
    sdf_metainfo = p.get_sdf_metainfo(sdf_file)
    ligand_names = sdf_metainfo["names"]
    structures = p.add_ligands_multi(pdb_file.split(".pdb")[0] + "_cleaned.pdb", "piascript_structures_tmp", ligands)
    result = PIA(structures, ligand_names = ligand_names, poses = "best", path = "current")
    p = result.plot("Analysis of SDF ligands", filename = output_name_prefix + "_analysis.png")
    r = result.save(output_name_prefix + "_analysis", True, True)
    c = result.to_csv(output_name_prefix + "_analysis.csv")

    # cleanup
    shutil.rmtree("piascript_structures_tmp")
    os.remove(pdb_file.split(".pdb")[0] + "_cleaned.pdb")

    return result

# workflow compare - input mode 1: one/two sdf file(s)
def compare(pdb_file, sdf_file_1, sdf_file_2 = None):

    """
    -- DESCRIPTION --
    Compare active and inactive complexes stored in a SDF file.
    """

    # create necessary directories
    structures_directory = "piascript_structures_tmp"
    structures_path = os.path.join(os.getcwd(), structures_directory)
    os.mkdir(structures_path)

    # read files
    filename = sdf_file_1
    if sdf_file_2 is not None:
        with open(sdf_file_1, "r") as f:
            actives = f.read()
            f.close()
        with open(sdf_file_2, "r") as f:
            inactives = f.read()
            f.close()
        content = actives + inactives
        with open("compare_combo_tmp.sdf", "w") as f:
            f.write(content)
            f.close()
        filename = "compare_combo_tmp.sdf"

    # extract interactions and frequencies
    output_name_prefix = sdf_file.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    p = Preparation()
    pdb = p.remove_ligands(pdb_file, pdb_file.split(".pdb")[0] + "_cleaned.pdb")
    ligands = p.get_ligands(filename)
    sdf_metainfo = p.get_sdf_metainfo(filename)
    ligand_names = sdf_metainfo["names"]
    structures = p.add_ligands_multi(pdb_file.split(".pdb")[0] + "_cleaned.pdb", "piascript_structures_tmp", ligands)
    actives_idx, inactives_idx = p.actives_inactives_split(filename)
    actives_structures = [structures[i] for i in actives_idx]
    actives_names = [ligand_names[i] for i in actives_idx]
    inactives_structures = [structures[i] for i in inactives_idx]
    inactives_names = [ligand_names[i] for i in inactives_idx]

    # actives
    result_1 = PIA(actives_structures, ligand_names = actives_names, poses = "best", path = "current")
    p_1 = result.plot("Active complexes", filename = output_name_prefix + "_actives_analysis.png")
    r_1 = result.save(output_name_prefix + "_actives_analysis", True, True)
    c_1 = result.to_csv(output_name_prefix + "_actives_analysis.csv")

    # inactives
    result_2 = PIA(inactives_structures, ligand_names = inactives_names, poses = "best", path = "current")
    p_2 = result.plot("Inactive complexes", filename = output_name_prefix + "_inactives_analysis.png")
    r_2 = result.save(output_name_prefix + "_inactives_analysis", True, True)
    c_2 = result.to_csv(output_name_prefix + "_inactives_analysis.csv")

    # actives vs inactives
    comparison = Comparison("Actives", "Inactives", result_1.i_frequencies, result_2.i_frequencies)
    p_3 = comparison.plot("Comparison: Actives vs. Inactives", filename = output_name_prefix + "_comparison.png")

    # cleanup
    shutil.rmtree("piascript_structures_tmp")
    os.remove(pdb_file.split(".pdb")[0] + "_cleaned.pdb")
    if sdf_file_2 is not None:
        os.remove("compare_combo_tmp.sdf")

    return [result_1, result_2, comparison]

# workflow score - input mode 1: one/two sdf file(s)
def score(pdb_file, sdf_file_1, sdf_file_2 = None):

    """
    -- DESCRIPTION --
    Train a scoring model from one or two SDF files. All results including model
    files are saved in the current directory.
    """

    # output file prefix
    output_name_prefix = sdf_file.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    # train model
    model = PIAModel()
    train_results = model.train(pdb_file, sdf_file_1, sdf_file_2, plot_prefix = output_name_prefix, keep_files = True)

    # save plots - ROC
    p_1 = plot_ROC_curve(train_results["TRAIN"]["+"]["ROC"]["fpr"], train_results["TRAIN"]["+"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_train_strat_p.png")
    p_2 = plot_ROC_curve(train_results["TRAIN"]["++"]["ROC"]["fpr"], train_results["TRAIN"]["++"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_train_strat_pp.png")
    p_3 = plot_ROC_curve(train_results["TRAIN"]["+-"]["ROC"]["fpr"], train_results["TRAIN"]["+-"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_train_strat_pm.png")
    p_4 = plot_ROC_curve(train_results["TRAIN"]["++--"]["ROC"]["fpr"], train_results["TRAIN"]["++--"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_train_strat_ppmm.png")
    p_5 = plot_ROC_curve(train_results["VAL"]["+"]["ROC"]["fpr"], train_results["VAL"]["+"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_val_strat_p.png")
    p_6 = plot_ROC_curve(train_results["VAL"]["++"]["ROC"]["fpr"], train_results["VAL"]["++"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_val_strat_pp.png")
    p_7 = plot_ROC_curve(train_results["VAL"]["+-"]["ROC"]["fpr"], train_results["VAL"]["+-"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_val_strat_pm.png")
    p_8 = plot_ROC_curve(train_results["VAL"]["++--"]["ROC"]["fpr"], train_results["VAL"]["++--"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_val_strat_ppmm.png")
    p_9 = plot_ROC_curve(train_results["TEST"]["+"]["ROC"]["fpr"], train_results["TEST"]["+"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_test_strat_p.png")
    p_10 = plot_ROC_curve(train_results["TEST"]["++"]["ROC"]["fpr"], train_results["TEST"]["++"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_test_strat_pp.png")
    p_11 = plot_ROC_curve(train_results["TEST"]["+-"]["ROC"]["fpr"], train_results["TEST"]["+-"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_test_strat_pm.png")
    p_12 = plot_ROC_curve(train_results["TEST"]["++--"]["ROC"]["fpr"], train_results["TEST"]["++--"]["ROC"]["tpr"],
                         filename = output_name_prefix + "_roc_test_strat_ppmm.png")

    # save plots - CM
    cm_1 = plot_confusion_matrix(train_results["TRAIN"]["+"]["CM"], [0, 1], filename = output_name_prefix + "_cm_train_strat_p.png")
    cm_2 = plot_confusion_matrix(train_results["TRAIN"]["++"]["CM"], [0, 1], filename = output_name_prefix + "_cm_train_strat_pp.png")
    cm_3 = plot_confusion_matrix(train_results["TRAIN"]["+-"]["CM"], [0, 1], filename = output_name_prefix + "_cm_train_strat_pm.png")
    cm_4 = plot_confusion_matrix(train_results["TRAIN"]["++--"]["CM"], [0, 1], filename = output_name_prefix + "_cm_train_strat_ppmm.png")
    cm_5 = plot_confusion_matrix(train_results["VAL"]["+"]["CM"], [0, 1], filename = output_name_prefix + "_cm_val_strat_p.png")
    cm_6 = plot_confusion_matrix(train_results["VAL"]["++"]["CM"], [0, 1], filename = output_name_prefix + "_cm_val_strat_pp.png")
    cm_7 = plot_confusion_matrix(train_results["VAL"]["+-"]["CM"], [0, 1], filename = output_name_prefix + "_cm_val_strat_pm.png")
    cm_8 = plot_confusion_matrix(train_results["VAL"]["++--"]["CM"], [0, 1], filename = output_name_prefix + "_cm_val_strat_ppmm.png")
    cm_9 = plot_confusion_matrix(train_results["TEST"]["+"]["CM"], [0, 1], filename = output_name_prefix + "_cm_test_strat_p.png")
    cm_10 = plot_confusion_matrix(train_results["TEST"]["++"]["CM"], [0, 1], filename = output_name_prefix + "_cm_test_strat_pp.png")
    cm_11 = plot_confusion_matrix(train_results["TEST"]["+-"]["CM"], [0, 1], filename = output_name_prefix + "_cm_test_strat_pm.png")
    cm_12 = plot_confusion_matrix(train_results["TEST"]["++--"]["CM"], [0, 1], filename = output_name_prefix + "_cm_test_strat_ppmm.png")

    # print and save summary statistics
    model.summary(filename = output_name_prefix + "_summary.txt")

    # save models
    model.save(output_name_prefix + "_best")
    model.change_strategy("+")
    model.save(output_name_prefix + "_p")
    model.change_strategy("++")
    model.save(output_name_prefix + "_pp")
    model.change_strategy("+-")
    model.save(output_name_prefix + "_pm")
    model.change_strategy("++--")
    model.save(output_name_prefix + "_ppmm")
    model.change_strategy("best")

    return model

# workflow predict - input mode 1: pdb file
def predict_pdb(model_info, pdb_file, cutoff = None):

    """
    -- DESCRIPTION --
    Predict a complex in PDB format.
    """

    # check if model or interactions are given
    if isinstance(model_info, str):
        model = PIAModel(filename = model_info)
    else:
        if cutoff is not None:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = cutoff)
        else:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = math.ceil(len(model_info)/2))

    # get and print prediction
    prediction = model.predict_pdb(pdb_file)
    print(prediction)

    return prediction

# workflow predict - input mode 2: sdf file
def predict_sdf(model_info, pdb_file, sdf_file, cutoff = None):

    """
    -- DESCRIPTION --
    Predict multiple, docked protein-ligand complexes from PDB + SDF files.
    """

    # check if model or interactions are given
    if isinstance(model_info, str):
        model = PIAModel(filename = model_info)
    else:
        if cutoff is not None:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = cutoff)
        else:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = math.ceil(len(model_info)/2))

    # return predicted dataset and save it as csv
    return model.predict_sdf(pdb_file, sdf_file, save_csv = True)

#### -------------------------------- MAIN -------------------------------- ####

# main function
def main():

    """
    -- DESCRIPTION --
    Main script checking for arguments and executing workflows.
    """

    # possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode",
                        choices = ("extract", "compare", "score", "predict"),
                        required = True,
                        dest = "mode",
                        help = "which workflow to run",
                        type = str
                        )
    parser.add_argument("-f", "--file",
                        required = True,
                        dest = "files",
                        help = "files to process",
                        type = str,
                        nargs = "+"
                        )
    parser.add_argument("-c", "--cutoff",
                        dest = "cutoff",
                        help = "cutoff for scoring model",
                        type = int
                        )
    args = parser.parse_args()

    # get supplied files
    files_dict = file_parser(args.files)

    # choose appropriate workflow based on mode
    if args.mode == "extract":
        if files_dict["sdf1"] is not None:
            if files_dict["pdb"] is not None:
                r = extract_sdf(files_dict["pdb"], files_dict["sdf1"])
            else:
                print("ERROR: PDB file is required but none was provided. Exiting!")
                r = 1
        else:
            if files_dict["txt"] is not None:
                pdb_codes = txt_to_list(files_dict["txt"])
                files_exist = True
                for code in pdb_codes:
                    if not os.path.isfile(pdb_codes[0]):
                        files_exist = False
                if files_exist:
                    r = extract_pdbs(pdb_codes)
                else:
                    r = extract_codes(pdb_codes)
            else:
                print("ERROR: TXT file of PDB codes or structures is required but none was provided. Exiting!")
                r = 1
    elif args.mode == "compare":
        if files_dict["pdb"] is not None and files_dict["sdf1"] is not None and files_dict["sdf2"] is not None:
            r = compare(files_dict["pdb"], files_dict["sdf1"], files_dict["sdf2"])
        else:
            print("ERROR: PDB file and SDF file are required but at least one of them was not provided. Exiting!")
            r = 1
    elif args.mode == "score":
        if files_dict["pdb"] is not None and files_dict["sdf1"] is not None and files_dict["sdf2"] is not None:
            r = score(files_dict["pdb"], files_dict["sdf1"], files_dict["sdf2"])
        else:
            print("ERROR: PDB file and SDF file are required but at least one of them was not provided. Exiting!")
            r = 1
    elif args.mode == "predict":
        if files_dict["pdb"] is not None:
            if files_dict["sdf1"] is not None:
                if files_dict["piam"] is not None:
                    r = predict_sdf(files_dict["piam"], files_dict["pdb"], files_dict["sdf1"])
                else:
                    if files_dict["txt"] is not None:
                        r = predict_sdf(txt_to_list(files_dict["txt"]), files_dict["pdb"], files_dict["sdf1"], cutoff = args.cutoff)
                    else:
                        print("ERROR: Model file or TXT file of interactions is required but none was provided. Exiting!")
                        r = 1
            else:
                if files_dict["piam"] is not None:
                    r = predict_pdb(files_dict["piam"], files_dict["pdb"])
                else:
                    if files_dict["txt"] is not None:
                        r = predict_pdb(txt_to_list(files_dict["txt"]), files_dict["pdb"], cutoff = args.cutoff)
                    else:
                        print("ERROR: Model file or TXT file of interactions is required but none was provided. Exiting!")
                        r = 1
        else:
            print("ERROR: PDB file is required but none was provided. Exiting!")
            r = 1
    else:
        r = 1

    # return worklow output or 1 if something went wrong
    return r

#### ------------------------------- SCRIPT ------------------------------- ####

# run main function
if __name__ == '__main__':
    r = main()
