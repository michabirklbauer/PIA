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
specifying the according arguments.

NOTES - TO BE DELETED!!!
https://docs.python.org/3/library/argparse.html
https://docs.python.org/3/howto/argparse.html
https://www.golinuxcloud.com/python-argparse/

ARGS:
-h --help: help
-m --mode: workflow mode
-f --file: files
-c --cutoff: cutoff
"""

import os
import math
import shutil
import argparse
import urllib.request as ur
from datetime import datetime
from PIAScore import *
from PIA import PIA as PIA
from PIA import Preparation as Preparation
from PIA import Comparison as Comparison
from PIAModel import PIAModel as PIAModel

# helper functions

## get elements from a comma delimited txt file
def txt_to_list(txt_file):

    """
    -- DESCRIPTION --
    """

    with open(txt_file, "r", encoding = "utf-8") as f:
        data = f.read()
        f.close()

    return [i.strip() for i in data.split(",")]

#
def file_parser(list_of_files):

    """
    -- DESCRIPTION --
    """

    pdb = None
    sdf1 = None
    sdf2 = None
    txt = None
    piam = None

    for f in list_of_files:
        if f.split(".")[-1] == "pdb":
            if pdb == None:
                pdb = f
                continue
            else:
                continue
        if f.split(".")[-1] == "sdf":
            if sdf1 == None:
                sdf1 = f
                continue
            elif sdf2 == None:
                sdf2 = f
                continue
            else:
                continue
        if f.split(".")[-1] == "txt":
            if txt == None:
                txt = f
                continue
            else:
                continue
        if f.split(".")[-1] == "piam":
            if piam == None:
                piam = f
                continue
            else:
                continue

    return {"pdb": pdb, "sdf1": sdf1, "sdf2": sdf2, "txt": txt, "piam": piam}


# extract interactions

## input mode 1: pdb codes
def extract_codes(list_of_codes):

    """
    -- DESCRIPTION --
    """

    filenames = [i + ".pdb" if i.split(".")[-1] != "pdb" else i for i in codes]
    download_links = ["https://files.rcsb.org/download/" + i for i in filenames]

    for i, link in enumerate(download_links):
        ur.urlretrieve(link, filenames[i])

    output_name_prefix = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    result = PIA(filenames)
    p = result.plot("Analysis of PDB Codes", filename = output_name_prefix + "_analysis.png")
    r = result.save(output_name_prefix + "_analysis", True)
    c = result.to_csv(output_name_prefix + "_analysis.csv")

    # cleanup
    for f in filenames:
        os.remove(f)

    return result

## input mode 2: pdb files
def extract_pdbs(list_of_files):

    """
    -- DESCRIPTION --
    """

    output_name_prefix = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    result = PIA(list_of_files)
    p = result.plot("Analysis of PDB files", filename = output_name_prefix + "_analysis.png")
    r = result.save(output_name_prefix + "_analysis", True)
    c = result.to_csv(output_name_prefix + "_analysis.csv")

    return result

## input mode 3: sdf file
def extract_sdf(pdb_file, sdf_file):

    """
    -- DESCRIPTION --
    """

    # create necessary directories
    structures_directory = "piascript_structures_tmp"
    structures_path = os.path.join(os.getcwd(), structures_directory)
    os.mkdir(structures_path)

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

# compare interactions

## input mode 1: one sdf file
## input mode 2: two sdf files
def compare(pdb_file, sdf_file_1, sdf_file_2 = None):

    """
    -- DESCRIPTION --
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

# scoring

## input mode 1: one sdf file
## input mode 2: two sdf files
def score(pdb_file, sdf_file_1, sdf_file_2 = None):

    """
    -- DESCRIPTION --
    """

    output_name_prefix = sdf_file.split(".sdf")[0] + datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    model = PIAModel()
    train_results = model.train(pdb_file, sdf_file_1, sdf_file_2, plot_prefix = output_name_prefix, keep_files = True)
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
    model.summary()
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

# prediction

## model input (either model file or supply params -> actives.txt, inactives.txt)

## input mode 1: pdb file
def predict_pdb(model_info, pdb_file, cutoff = None):

    """
    -- DESCRIPTION --
    """

    if isinstance(model_info, str):
        model = PIAModel(filename = model_info)
    else:
        if cutoff is not None:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = cutoff)
        else:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = math.ceil(len(model_info)/2))

    prediction = model.predict_pdb(pdb_file)
    print(prediction)

    return prediction

## input mode 2: sdf file
def predict_sdf(model_info, pdb_file, sdf_file, cutoff = None):

    """
    -- DESCRIPTION --
    """

    if isinstance(model_info, str):
        model = PIAModel(filename = model_info)
    else:
        if cutoff is not None:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = cutoff)
        else:
            model = PIAModel(positives = model_info, strategy = "+", cutoff = math.ceil(len(model_info)/2))

    return model.predict_sdf(pdb_file, sdf_file, save_csv = True)

# main function of the script
def main():

    """
    -- DESCRIPTION --
    """

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

    files_dict = file_parser(args.files)

    if args.mode == "extract":
        pass
    elif args.mode == "compare":
        pass
    elif args.mode == "score":
        pass
    elif args.mode == "predict":
        pass
    else
        pass

if __name__ == '__main__':
    r = main()
