"""
Original script credited to:
Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
Apache 2.0 License.
"""

import argparse
import os
import shutil
from pathlib import Path

from z_ssmnet.ssl_read_data_from_disk.pretrain.ssl_mnet_zonal import pretrain


def safe_copytree(src: Path, dst: Path):
    """
    Cross-platform safe version of shutil.copytree().
    Allows overwriting dst if it already exists (Python >=3.8).
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Pretrain nnU-Net model (cross-platform).")

    parser.add_argument("--preprocesseddir", type=str, default=os.environ.get("SM_CHANNEL_PREPROCESSED", "./preprocessed"), help="Directory containing preprocessed input data.")
    parser.add_argument("--outputdir", type=str, default=os.environ.get("SM_MODEL_DIR", "./output"), help="Directory to save pretrained model.")
    parser.add_argument( "--checkpointsdir", type=str, default="./checkpoints", help="Directory where checkpoints will be stored.")

    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocesseddir)
    output_dir = Path(args.outputdir)
    checkpoints_dir = Path(args.checkpointsdir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== PRETRAIN CONFIGURATION ===")
    print(f"checkpoints_dir: {checkpoints_dir.resolve()}")
    print(f"preprocessed_dir: {preprocessed_dir.resolve()}")
    print(f"output_dir:       {output_dir.resolve()}")
    print("================================\n")

    # Check required directories
    cubes_dir = preprocessed_dir / "SSL" / "generated_cubes"
    if not cubes_dir.exists():
        raise FileNotFoundError(f"Generated cubes not found: {cubes_dir}")

    # Where pretrained weights will be written
    model_out_dir = checkpoints_dir / "SSL" / "pretrained_weights"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Run pretraining
    print(">> Running SSL pretraining...")
    pretrain(
        model_dir=model_out_dir,
        data_dir=cubes_dir,
    )
    print(">> Pretraining complete.\n")

    # Export trained model
    print(">> Exporting pretrained model...")
    final_export = output_dir / "SSL" / "pretrained_weights"
    final_export.parent.mkdir(parents=True, exist_ok=True)

    safe_copytree(model_out_dir, final_export)

    print(f">> Model exported to: {final_export.resolve()}")
    print("DONE.")


if __name__ == "__main__":
    main()
