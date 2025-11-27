# This code is adapted from https://github.com/MrGiovanni/ModelsGenesis/blob/master/infinite_generator_3D.py. 
# The original code is licensed under the attached LICENSE (https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/ssl/LICENSE).

import warnings
warnings.filterwarnings('ignore')

import sys
import random
from pathlib import Path
import re
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse
from glob import glob
from setup_config import setup_config

def safe_normalize(arr):
    """Normalize array to [0,1], tolerant to constant arrays."""
    amin = np.min(arr)
    amax = np.max(arr)
    if amax - amin == 0:
        return np.zeros_like(arr, dtype=float)
    return 1.0 * (arr - amin) / (amax - amin)

def make_cubes(
    fold: int = 0,
    input_rows: int = 64,
    input_cols: int = 64,
    input_deps: int = 16,
    crop_rows: int = 64,
    crop_cols: int = 64,
    data_dir: str = None,
    save_dir: str = None,
    scale: int = 12,
):
    sys.setrecursionlimit(40000)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    if data_dir is None:
        raise ValueError("data_dir must be provided")
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"data_dir is not a valid directory: {data_dir}")

    if save_dir is None:
        raise ValueError("save_dir must be provided")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if fold < 0 or fold > 5:
        raise ValueError(f"fold must be in [0, 5]: {fold}")

    config = setup_config(input_rows=input_rows,
        input_cols=input_cols,
        input_deps=input_deps,
        crop_rows=crop_rows,
        crop_cols=crop_cols,
        scale=scale,
        len_border=0,
        len_border_z=0,
        DATA_DIR=str(data_dir),
        )
    config.display()

    def infinite_generator_from_one_volume(config, t2_array, adc_array, dwi_array, seg_array):
        size_x, size_y, size_z = t2_array.shape

        # Normalise channels safely
        t2_array = safe_normalize(t2_array.astype(float))
        adc_clipped = np.clip(adc_array.astype(float), config.adc_min, config.adc_max)
        adc_array = 1.0 * (adc_clipped - config.adc_min) / max((config.adc_max - config.adc_min), 1.0)
        dwi_array = safe_normalize(dwi_array.astype(float))

        # Pre-allocate storage for one "batch" of crops (scale)
        slice_set = np.zeros((config.scale, 4, config.input_rows, config.input_cols, config.input_deps), dtype=float)

        num_pair = 0
        cnt = 0
        max_tries = 50 * config.scale
        while True:
            cnt += 1
            if cnt > max_tries and num_pair == 0:
                return None
            elif cnt > max_tries and num_pair > 0:
                return np.array(slice_set[:num_pair])

            start_x = random.randint(0 + config.len_border, max(0, size_x - config.crop_rows - 1 - config.len_border))
            start_y = random.randint(0 + config.len_border, max(0, size_y - config.crop_cols - 1 - config.len_border))
            start_z = random.randint(0 + config.len_border_z, max(0, size_z - config.input_deps - 1 - config.len_border_z))

            # If the volume is smaller than crop, skip
            if start_x + config.crop_rows > size_x or start_y + config.crop_cols > size_y or start_z + config.input_deps > size_z:
                continue

            t2_crop_window = t2_array[start_x : start_x + config.crop_rows,
                                     start_y : start_y + config.crop_cols,
                                     start_z : start_z + config.input_deps]

            adc_crop_window = adc_array[start_x : start_x + config.crop_rows,
                                        start_y : start_y + config.crop_cols,
                                        start_z : start_z + config.input_deps]

            dwi_crop_window = dwi_array[start_x : start_x + config.crop_rows,
                                        start_y : start_y + config.crop_cols,
                                        start_z : start_z + config.input_deps]

            seg_crop_window = seg_array[start_x : start_x + config.crop_rows,
                                        start_y : start_y + config.crop_cols,
                                        start_z : start_z + config.input_deps]

            # Ensure shapes match expectations
            if (t2_crop_window.shape != (config.crop_rows, config.crop_cols, config.input_deps) or
                adc_crop_window.shape != (config.crop_rows, config.crop_cols, config.input_deps) or
                dwi_crop_window.shape != (config.crop_rows, config.crop_cols, config.input_deps) or
                seg_crop_window.shape != (config.crop_rows, config.crop_cols, config.input_deps)):
                # skip bad crop
                continue

            crop_window = np.stack((t2_crop_window, adc_crop_window, dwi_crop_window, seg_crop_window), axis=0)
            slice_set[num_pair] = crop_window

            num_pair += 1
            if num_pair == config.scale:
                break

        return np.array(slice_set)

    def get_self_learning_data(fold_index, config):
        slice_set = []
        # Expect subdirectories like subset0 ... subset5
        picai_subset_path = Path(config.DATA_DIR) / f"subset{fold_index}"
        if not picai_subset_path.exists():
            print(f"Warning: subset directory not found: {picai_subset_path}")
            return np.array(slice_set)

        # Look for *_0000.nii or *_0000.nii.gz
        file_list = sorted(picai_subset_path.glob("*_0000.nii*"))
        for img_path in tqdm(file_list, desc=f"Reading subset{fold_index}"):
            img_file = str(img_path)
            print(img_file)

            try:
                t2_img = sitk.ReadImage(img_file, sitk.sitkFloat32)
            except Exception as e:
                print(f"Failed reading {img_file}: {e}")
                continue

            t2_array = sitk.GetArrayFromImage(t2_img)  # SITK returns z,y,x
            # Transpose to x,y,z to match original script's expectation
            t2_array = t2_array.transpose(2, 1, 0)

            # Build related filenames robustly:
            # Replace trailing _0000 before .nii or .nii.gz
            basename = img_path.name
            # handle both .nii and .nii.gz
            seg_name = re.sub(r'_0000(\.nii(\.gz)?)$', r'\1', basename)
            adc_name = basename.replace('_0000', '_0001')
            dwi_name = basename.replace('_0000', '_0002')

            adc_path = img_path.with_name(adc_name)
            dwi_path = img_path.with_name(dwi_name)
            seg_path = img_path.with_name(seg_name)

            if not adc_path.exists():
                print(f"Missing ADC file: {adc_path} (skipping)")
                continue
            if not dwi_path.exists():
                print(f"Missing DWI file: {dwi_path} (skipping)")
                continue
            if not seg_path.exists():
                print(f"Missing SEG file: {seg_path} (skipping)")
                continue

            try:
                adc_img = sitk.ReadImage(str(adc_path), sitk.sitkFloat32)
                adc_array = sitk.GetArrayFromImage(adc_img).transpose(2, 1, 0)

                dwi_img = sitk.ReadImage(str(dwi_path), sitk.sitkFloat32)
                dwi_array = sitk.GetArrayFromImage(dwi_img).transpose(2, 1, 0)

                seg_img = sitk.ReadImage(str(seg_path), sitk.sitkFloat32)
                seg_array = sitk.GetArrayFromImage(seg_img).transpose(2, 1, 0)
            except Exception as e:
                print(f"Failed to read related files for {img_file}: {e}")
                continue

            x = infinite_generator_from_one_volume(config, t2_array, adc_array, dwi_array, seg_array)
            if x is not None:
                slice_set.extend(x)

        if len(slice_set) == 0:
            return np.array(slice_set)
        return np.array(slice_set)

    print(">> Fold {}".format(fold))
    cube = get_self_learning_data(fold, config)
    if cube.size == 0:
        print("No cubes generated; output will be an empty array.")
    else:
        print("cube: {} | {:.6f} ~ {:.6f}".format(cube.shape, np.min(cube), np.max(cube)))
    out_name = save_dir / f"bat_{config.scale}_s_{config.input_rows}x{config.input_cols}x{config.input_deps}_{fold}.npy"
    np.save(str(out_name), cube)
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D training cubes from PICAI-style folders (cross-platform)")
    parser.add_argument("--fold", type=int, required=True, help="fold of subset (0-5)")
    parser.add_argument("--input_rows", type=int, default=64)
    parser.add_argument("--input_cols", type=int, default=64)
    parser.add_argument("--input_deps", type=int, default=16)
    parser.add_argument("--crop_rows", type=int, default=64)
    parser.add_argument("--crop_cols", type=int, default=64)
    parser.add_argument("--data", type=str, required=True, help="directory containing subset0..subset5 folders")
    parser.add_argument("--save", type=str, required=True, help="output directory for generated cubes")
    parser.add_argument("--scale", type=int, default=12)
    args = parser.parse_args()

    make_cubes(
        fold=args.fold,
        input_rows=args.input_rows,
        input_cols=args.input_cols,
        input_deps=args.input_deps,
        crop_rows=args.crop_rows,
        crop_cols=args.crop_cols,
        data_dir=args.data,
        save_dir=args.save,
        scale=args.scale,
    )
