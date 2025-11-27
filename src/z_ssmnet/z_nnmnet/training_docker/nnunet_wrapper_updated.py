#!/usr/bin/env python3
"""
Windows-compatible nnUNet pipeline wrapper.
"""

import argparse
import functools
import os
import pickle
import re
import subprocess
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from carbontracker.tracker import CarbonTracker
from customerized_carbon_tracker import CustomizedCarbonTracker

# Replace shutil_sol (not available on Windows)
# nnUNet uses a slightly modified shutil, but standard shutil works here
import shutil as shutil_sol

from io import checksum, path_exists, read_json, refresh_file_list, write_json
from picai_prep.data_utils import atomic_file_copy

PLANS = "nnUNetPlansv2.1"

print = functools.partial(print, flush=True)

def get_task_id(task_name):
    return re.match(r"Task([0-9]+)", task_name).group(1)


def print_split_per_fold(split_file, fold=None):
    try:
        with split_file.open("rb") as pkl:
            splits = pickle.load(pkl)
    except FileNotFoundError:
        print("Split file not found")
    else:
        for i, split in enumerate(splits):
            if fold not in (None, i):
                continue

            print(f"Fold #{i}")
            print("> Training")
            for caseid in sorted(split["train"]):
                print(f">> {caseid}")
            print("> Validation")
            for caseid in sorted(split["val"]):
                print(f">> {caseid}")

            if i + 1 < len(splits):
                print("-" * 25)


def prepare(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("data", type=str)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--modality", type=str, default="CT")
    parser.add_argument("--labels", type=str, nargs="*", default=["background", "foreground"])
    parser.add_argument("--license", type=str, default="")
    parser.add_argument("--release", type=str, default="1.0")
    args = parser.parse_args(argv)

    print("[#] Creating directory structure")

    datadir = Path(args.data)
    taskdir = datadir / "nnUNet_raw_data" / args.task

    try:
        taskdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'Destination "{taskdir}" already exists')
        return

    # detect image patterns
    image_srcdir = Path(args.images)
    if "*" in image_srcdir.name:
        image_glob = image_srcdir.name
        image_srcdir = image_srcdir.parent
    else:
        image_glob = "*.mha"

    image_dstdir = taskdir / "imagesTr"
    image_dstdir.mkdir()

    mask_srcdir = Path(args.masks)
    mask_dstdir = taskdir / "labelsTr"
    mask_dstdir.mkdir()

    print("[#] Converting images and masks")
    training = []

    for image_srcfile in sorted(image_srcdir.glob(image_glob)):
        if image_srcfile.name.startswith("."):
            continue

        if image_srcfile.name.endswith(".nii.gz"):
            caseid = image_srcfile.name[:-7]
            ext = "nii.gz"
        else:
            caseid = image_srcfile.stem
            ext = image_srcfile.suffix[1:]

        if caseid.endswith("_0000"):
            caseid = caseid[:-5]

        try:
            mask_srcfile = mask_srcdir / f"{caseid}.{ext}"
            if not mask_srcfile.exists():
                mask_srcfile = next(mask_srcdir.glob(f"{caseid}_*.{ext}"))
        except StopIteration:
            print(f'Missing mask for case "{caseid}"')
            return

        image_dstfile = image_dstdir / f"{caseid}_0000.nii.gz"
        print(f"{image_srcfile.name} -> {image_dstfile.name}")
        atomic_file_copy(image_srcfile, image_dstfile)

        mask_dstfile = mask_dstdir / f"{caseid}.nii.gz"
        atomic_file_copy(mask_srcfile, mask_dstfile)

        training.append({
            "image": f"./imagesTr/{caseid}.nii.gz",
            "label": f"./labelsTr/{caseid}.nii.gz"
        })

    # dataset.json
    print("[#] Writing metadata to dataset.json")

    name = args.task.split("_", 1)[1]
    labels = OrderedDict([(str(i), label) for i, label in enumerate(args.labels)])

    metadata = OrderedDict([
        ("name", name),
        ("description", f"{name}, reformatted for nnUNet"),
        ("tensorImageSize", "3D"),
        ("licence", args.license),
        ("release", args.release),
        ("modality", {"0": args.modality}),
        ("labels", labels),
        ("numTraining", len(training)),
        ("numTest", 0),
        ("training", training),
        ("test", []),
    ])

    write_json(taskdir / "dataset.json", metadata, make_dirs=False)

def plan_train(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("data", type=str)
    parser.add_argument("--results", type=str, required=False)
    parser.add_argument("--prepdir", type=str, default="./nnUNet_preprocessed")
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2")
    parser.add_argument("--trainer_kwargs", required=False, default="{}")
    parser.add_argument("--kwargs", type=str, required=False, default=None)
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--custom_split", type=str)
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--ensembling", action="store_true")
    parser.add_argument("--use_compressed_data", action="store_true")
    parser.add_argument("--plan_2d", action="store_true")
    parser.add_argument("--dont_plan_3d", action="store_true")
    parser.add_argument("--carbontracker", action="store_true")
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--disable_validation_inference", action="store_true")
    parser.add_argument("--dont_copy_preprocessed_data", action="store_true")
    args = parser.parse_args(argv)

    args.task = str(args.task)

    datadir = Path(args.data)
    prepdir = Path(args.prepdir)
    splits_file = prepdir / args.task / "splits_final.pkl"

    os.environ["nnUNet_raw_data_base"] = str(datadir)
    os.environ["nnUNet_preprocessed"] = str(prepdir)
    os.environ["RESULTS_FOLDER"] = args.results or str(datadir / "results")

    # CarbonTracker
    with CustomizedCarbonTracker(prepdir / "carbontracker", enabled=args.carbontracker):

        taskid = get_task_id(args.task)
        taskdir = datadir / "nnUNet_preprocessed" / args.task

        if path_exists(taskdir) or path_exists(prepdir / args.task):
            print(f"[#] Found preprocessed data for {args.task}")

        else:
            print("[#] Running planning and preprocessing")
            cmd = [
                "nnUNet_plan_and_preprocess",
                "-t", taskid,
                "-tl", os.environ.get("nnUNet_tl", "8"),
                "-tf", os.environ.get("nnUNet_tf", "8"),
                "--verify_dataset_integrity",
            ]
            if not args.plan_2d and "2d" not in args.network:
                cmd.extend(["--planner2d", "None"])
            if args.dont_plan_3d and "3d" not in args.network:
                cmd.extend(["--planner3d", "None"])

            subprocess.check_call(cmd)

        if args.plan_only:
            return

        cmd = [
            "nnUNet_train",
            args.network,
            args.trainer,
            taskid,
            args.fold
        ]

        fold_name = "all" if args.fold == "all" else f"fold_{args.fold}"
        outdir = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet" / args.network / args.task / f"{args.trainer}__{PLANS}" / fold_name

        if args.validation_only:
            print("[#] Running validation only")
            cmd.append("--validation_only")
        elif path_exists(outdir) and any(outdir.glob("*.model")):
            print("[#] Resuming training")
            cmd.append("-c")
        elif args.pretrained_weights:
            print("[#] Loading pretrained weights")
            cmd.extend(["-pretrained_weights", args.pretrained_weights])
        else:
            print("[#] Starting training")

        if args.trainer_kwargs:
            cmd.append(f"--trainer_kwargs={args.trainer_kwargs}")
        if args.use_compressed_data:
            cmd.append("--use_compressed_data")
        if args.ensembling:
            cmd.append("--npz")
        if args.kwargs:
            cmd.extend(args.kwargs.split(" "))
        if args.disable_validation_inference:
            cmd.append("--disable_validation_inference")

        print(f"[#] Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

        # copy split
        if splits_file.exists() and splits_file.parent != taskdir and taskdir.exists():
            atomic_file_copy(splits_file, taskdir)


def reveal_split(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("data", type=str)
    args = parser.parse_args(argv)

    datadir = Path(args.data)
    split_file = datadir / "nnUNet_preprocessed" / args.task / "splits_final.pkl"

    print_split_per_fold(split_file)

def find_best_configuration(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("data", type=str)
    parser.add_argument("--networks", type=str, nargs="*", default=["3d_fullres"])
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2")
    args = parser.parse_args(argv)

    datadir = Path(args.data)
    prepdir = datadir / "nnUNet_preprocessed"
    os.environ["nnUNet_preprocessed"] = str(prepdir)
    os.environ["RESULTS_FOLDER"] = str(datadir / "results")

    print("[#] Preparing output directory")
    (datadir / "results" / "nnUNet" / "ensembles" / args.task).mkdir(parents=True, exist_ok=True)

    for network in args.networks:
        print(f"[#] Postprocessing for {network}")
        subprocess.check_call([
            "nnUNet_determine_postprocessing",
            "-m", network,
            "-t", get_task_id(args.task),
            "-tr", args.trainer
        ])

    if len(args.networks) > 1:
        print("[#] Finding best ensemble")
        refresh_file_list(prepdir / args.task / "gt_segmentations")
        subprocess.check_call([
            "nnUNet_find_best_configuration",
            "-m", *args.networks,
            "-t", get_task_id(args.task),
            "-tr", args.trainer
        ])


def _predict(args):
    os.environ["RESULTS_FOLDER"] = args.results

    outdir = Path(args.output).absolute()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNet_predict",
        "-t", args.task,
        "-i", args.input,
        "-o", args.output,
        "-m", args.network,
        "-tr", args.trainer,
        "--num_threads_preprocessing", "2",
        "--num_threads_nifti_save", "1"
    ]

    if args.folds:
        cmd.extend(["-f", *args.folds.split(",")])
    if args.checkpoint:
        cmd.extend(["-chk", args.checkpoint])
    if args.store_probability_maps:
        cmd.append("--save_npz")
    if args.disable_augmentation:
        cmd.append("--disable_tta")
    if args.disable_patch_overlap:
        cmd.extend(["--step_size", "1"])
    if args.lowres_segmentations:
        cmd.extend(["-l", args.lowres_segmentations])

    subprocess.check_call(cmd)


def predict(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--input", type=str, default="./input")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--network", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2")
    parser.add_argument("--folds", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--store_probability_maps", action="store_true")
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--disable_patch_overlap", action="store_true")
    parser.add_argument("--lowres_segmentations", type=str)
    args = parser.parse_args(argv)

    _predict(args)

def ensemble(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--input", type=str, default="./input")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--networks", type=str, nargs="*", default=["3d_fullres"])
    parser.add_argument("--trainers", type=str, nargs="*", default=["nnUNetTrainerV2"])
    parser.add_argument("--folds", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--disable_augmentation", action="store_true")
    parser.add_argument("--disable_patch_overlap", action="store_true")
    args = parser.parse_args(argv)

    output_dirs = []
    ensemble_frags = []

    for i, network in enumerate(args.networks):
        print(f"[#] Running inference for network: {network}")

        args_pred = deepcopy(args)
        args_pred.store_probability_maps = True
        args_pred.network = network
        args_pred.trainer = args.trainers[i] if i < len(args.trainers) else args.trainers[-1]
        del args_pred.networks
        del args_pred.trainers

        out_dir = Path(args.output) / network
        out_dir.mkdir(parents=True, exist_ok=True)
        args_pred.output = str(out_dir)

        ensemble_frags.append(f"{args_pred.network}__{args_pred.trainer}__{PLANS}")
        output_dirs.append(out_dir)

        _predict(args_pred)

    print("[#] Ensembling results")
    ensemble_name = "ensemble_" + "--".join(ensemble_frags)
    outdir = Path(args.output) / ensemble_name

    cmd = [
        "nnUNet_ensemble",
        "-f", *[str(f) for f in output_dirs],
        "-o", str(outdir)
    ]

    pp_file = Path(args.results) / "nnUNet" / "ensembles" / args.task / ensemble_name / "postprocessing.json"
    if path_exists(pp_file):
        cmd.extend(["-pp", str(pp_file)])

    subprocess.check_call(cmd)

def evaluate(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    args = parser.parse_args(argv)

    gt_dir = Path(args.ground_truth)
    pred_dir = Path(args.prediction)

    if not path_exists(gt_dir):
        print("Ground-truth folder does not exist")
        return
    if not path_exists(pred_dir):
        print("Prediction folder does not exist")
        return

    # label ranges
    range_re = re.compile(r"[0-9]+-[0-9]+")
    if len(args.labels) == 1 and range_re.fullmatch(args.labels[0]):
        lo, hi = map(int, args.labels[0].split("-"))
        labels = [str(x) for x in range(lo, hi + 1)]
    else:
        labels = args.labels

    print("[#] Running evaluation")
    subprocess.check_call([
        "nnUNet_evaluate_folder",
        "-ref", str(gt_dir),
        "-pred", str(pred_dir),
        "-l", *labels
    ])

    results_file = pred_dir / "summary.json"
    try:
        results = read_json(results_file)
    except IOError:
        print("Evaluation failed")
        return

    print("Average Dice:")
    for label, metrics in sorted(results["results"]["mean"].items(), key=lambda x: int(x[0])):
        print(f" {label}: {metrics['Dice']}")

def checkout(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=str, default="")
    args, rest = parser.parse_known_args(argv)

    if args.checkout:
        repo = Path("./nnunet").absolute()
        if not repo.exists():
            print(f"[!] Repo {repo} does not exist (expected nnUNet directory).")
        else:
            subprocess.check_call(["git", "-C", str(repo), "fetch"])
            subprocess.check_call(["git", "-C", str(repo), "checkout", args.checkout])

    return rest

if __name__ == "__main__":
    actions = {
        "prepare": prepare,
        "plan_train": plan_train,
        "reveal_split": reveal_split,
        "find_best_configuration": find_best_configuration,
        "predict": predict,
        "ensemble": ensemble,
        "evaluate": evaluate,
    }

    try:
        action = actions[sys.argv[1]]
        argv = checkout(sys.argv[2:])
    except (IndexError, KeyError):
        print("Usage: nnunet " + " / ".join(actions.keys()) + " ...")
    else:
        action(argv)
