import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

COMMANDS = ("bam2vcf", "snv2barcode", "pre_train", "snv_effect", "get_template")


def _build_subcommand_parser(command: str) -> argparse.ArgumentParser:
    descriptions = {
        "bam2vcf": (
            "Run the Bash SNV-calling pipeline for one or more BAM files. "
            "The pipeline generates filtered VCF files after removing RNA-editing sites."
        ),
        "snv2barcode": (
            "Build per-sample and merged barcode-by-SNV AnnData matrices from "
            "BAM, VCF, and barcode inputs defined in a YAML configuration file."
        ),
        "pre_train": (
            "Align pretraining and finetuning RNA AnnData inputs, then train the "
            "RNA-only backbone model for the downstream SNV effect workflow."
        ),
        "snv_effect": (
            "Train or evaluate the SNV perturbation model using aligned RNA and "
            "barcode-by-SNV AnnData inputs, then export functional-effect results."
        ),
        "get_template": (
            "Write a train_config.yaml template into the current directory for use "
            "with the pre_train and snv_effect commands."
        ),
    }
    examples = {
        "bam2vcf": (
            "prismsnv bam2vcf --outer-jobs 6 --inner-threads 4 \\\n"
            "  --reference genome.fa --varscan-jar VarScan.jar \\\n"
            "  --rna-edit-bed RNA_edit.bed --out-dir ./snv_call_out \\\n"
            "  --bam-files sample1.bam sample2.bam"
        ),
        "snv2barcode": "prismsnv snv2barcode snv2barcode_config.yaml",
        "pre_train": "prismsnv pre_train -y train_config.yaml",
        "snv_effect": "prismsnv snv_effect --n_gpu 3 -y train_config.yaml",
        "get_template": "prismsnv get_template --output train_config.yaml",
    }
    parser = argparse.ArgumentParser(
        prog=f"prismsnv {command}",
        description=descriptions[command],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Example:\n  {examples[command]}",
    )

    if command == "bam2vcf":
        parser.add_argument(
            "--outer-jobs",
            required=True,
            metavar="N",
            help="Number of BAM files to process in parallel.",
        )
        parser.add_argument(
            "--inner-threads",
            required=True,
            metavar="N",
            help="Number of samtools threads used for each BAM file.",
        )
        parser.add_argument(
            "--reference",
            required=True,
            metavar="FASTA",
            help="Reference genome FASTA with a readable .fai index.",
        )
        parser.add_argument(
            "--varscan-jar",
            required=True,
            metavar="JAR",
            help="Path to VarScan.jar.",
        )
        parser.add_argument(
            "--rna-edit-bed",
            required=True,
            metavar="BED",
            help="BED file containing RNA-editing sites to remove.",
        )
        parser.add_argument(
            "--out-dir",
            required=True,
            metavar="DIR",
            help="Output directory for generated SNV-calling files.",
        )
        parser.add_argument(
            "--bam-files",
            required=True,
            nargs="+",
            metavar="BAM",
            help="One or more input BAM files.",
        )
    elif command == "snv2barcode":
        parser.add_argument(
            "config",
            metavar="CONFIG",
            help="YAML file defining BAM, VCF, barcode, and output settings.",
        )
    elif command == "get_template":
        parser.add_argument(
            "-o",
            "--output",
            dest="output",
            default="train_config.yaml",
            metavar="PATH",
            help="Destination path for the generated template (default: train_config.yaml).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite the destination file if it already exists.",
        )
    else:
        if command == "snv_effect":
            parser.add_argument(
                "--n_gpu",
                "--n-gpu",
                type=int,
                metavar="N",
                help="Number of GPUs used to launch distributed training.",
            )
        parser.add_argument(
            "-y",
            "--yaml",
            "--config",
            dest="yaml_path",
            metavar="CONFIG",
            help="Path to the YAML configuration file.",
        )
        parser.add_argument(
            "yaml_path_pos",
            nargs="?",
            metavar="CONFIG",
            help="Positional alternative for the YAML configuration file path.",
        )

    return parser


def _extract_snv_effect_launcher_args(command_args: list[str]) -> tuple[Optional[int], list[str]]:
    snv_args: list[str] = []
    n_gpu: Optional[int] = None
    skip_next = False
    for index, arg in enumerate(command_args):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--n_gpu", "--n-gpu"}:
            if index + 1 >= len(command_args):
                raise SystemExit("ERROR: --n_gpu requires an integer value.\n")
            try:
                n_gpu = int(command_args[index + 1])
            except ValueError as exc:
                raise SystemExit("ERROR: --n_gpu must be an integer greater than 0.\n") from exc
            skip_next = True
        elif arg.startswith("--n_gpu=") or arg.startswith("--n-gpu="):
            value = arg.split("=", 1)[1]
            try:
                n_gpu = int(value)
            except ValueError as exc:
                raise SystemExit("ERROR: --n_gpu must be an integer greater than 0.\n") from exc
        else:
            snv_args.append(arg)
    if n_gpu is not None and n_gpu <= 0:
        raise SystemExit("ERROR: --n_gpu must be an integer greater than 0.\n")
    return n_gpu, snv_args


def _already_under_torchrun() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ


def _run_snv_effect_with_optional_torchrun(command_args: list[str]) -> None:
    n_gpu, snv_args = _extract_snv_effect_launcher_args(command_args)
    should_launch_distributed = (
        "-h" not in snv_args
        and "--help" not in snv_args
        and n_gpu is not None
        and n_gpu > 1
        and not _already_under_torchrun()
    )
    if should_launch_distributed:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={n_gpu}",
                "-m",
                "prismsnv.cli",
                "snv_effect",
                *snv_args,
            ]
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    from .train.snv_effect import main as snv_effect_main

    snv_effect_main(snv_args)


def _write_train_config_template(
    parser: argparse.ArgumentParser, output: str, force: bool
) -> None:
    from importlib import resources

    destination = Path(output)
    if destination.exists() and not force:
        parser.exit(
            1,
            f"ERROR: {destination} already exists. Use --force to overwrite it.\n",
        )

    try:
        template_text = (
            resources.files("prismsnv.templates")
            .joinpath("train_config.yaml")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        parser.exit(1, f"ERROR: Cannot locate the bundled train_config.yaml template: {exc}\n")

    parent = destination.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(template_text, encoding="utf-8")
    print(f"Wrote train_config.yaml template to {destination}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    parser = argparse.ArgumentParser(
        prog="prismsnv",
        description="PrismSNV command line tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Subcommands:
              bam2vcf
                Runs the Bash SNV-calling pipeline; requires bash, samtools,
                java, bedtools, and awk.
                Example:
                  prismsnv bam2vcf --outer-jobs 6 --inner-threads 4 \\
                    --reference genome.fa --varscan-jar VarScan.jar \\
                    --rna-edit-bed RNA_edit.bed --out-dir ./snv_call_out \\
                    --bam-files sample1.bam sample2.bam

              snv2barcode
                Builds per-sample and merged barcode x SNV AnnData matrices
                from BAM/VCF/barcode inputs defined in a YAML config.
                Example:
                  prismsnv snv2barcode <snv2barcode_config.yaml>

              pre_train
                Aligns RNA AnnData inputs, trains the RNA backbone, and writes
                finetune_aligned.h5ad plus the backbone checkpoint.
                Example:
                  prismsnv pre_train -y <train_config.yaml>

              snv_effect
                Trains or evaluates the SNV perturbation model, then exports
                attention, perturbation score tables, and downstream plots.
                Example:
                  prismsnv snv_effect --n_gpu 3 -y <train_config.yaml>

              get_template
                Writes a train_config.yaml template into the current directory
                for the pre_train and snv_effect commands.
                Example:
                  prismsnv get_template --output train_config.yaml

            Typical workflow:
              0. prismsnv get_template
              1. prismsnv bam2vcf ...
              2. prismsnv snv2barcode <snv2barcode_config.yaml>
              3. prismsnv pre_train -y <train_config.yaml>
              4. prismsnv snv_effect --n_gpu 3 -y <train_config.yaml>

            Use 'prismsnv <subcommand> --help' for subcommand-specific options.
            """
        ),
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="Pipeline stage to run.",
    )
    if not argv:
        parser.print_help()
        parser.exit(2, "\nerror: the following arguments are required: command\n")
    if argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = argv[0]
    command_args = argv[1:]
    if command not in COMMANDS:
        parser.error(
            f"argument command: invalid choice: {command!r} "
            f"(choose from {', '.join(repr(item) for item in COMMANDS)})"
        )
    if any(arg in {"-h", "--help"} for arg in command_args):
        _build_subcommand_parser(command).parse_args(command_args)
        return

    if command == "pre_train":
        from .train.pre_train import main as pre_train_main

        pre_train_main(command_args)
    elif command == "snv_effect":
        _run_snv_effect_with_optional_torchrun(command_args)
    elif command == "get_template":
        template_args = _build_subcommand_parser(command).parse_args(command_args)
        _write_train_config_template(parser, template_args.output, template_args.force)
    elif command == "snv2barcode":
        snv2barcode_args = _build_subcommand_parser(command).parse_args(command_args)

        from .preprocess.snv2barcode import main as snv2barcode_main

        snv2barcode_main(snv2barcode_args.config)
    elif command == "bam2vcf":
        from .preprocess import __file__ as preprocess_init

        script_path = Path(preprocess_init).with_name("bam2vcf.sh")
        if not script_path.is_file():
            parser.exit(1, f"ERROR: Cannot find bundled bam2vcf.sh at {script_path}\n")

        bash_path = shutil.which("bash")
        if bash_path is None:
            parser.exit(
                127,
                "ERROR: 'bash' was not found in PATH. "
                "Install Git Bash, WSL, or another Bash runtime before running "
                "prismsnv bam2vcf.\n",
            )

        script_text = script_path.read_text(encoding="utf-8")
        script_text = script_text.replace("\r\n", "\n").replace("\r", "\n")
        completed = subprocess.run(
            [bash_path, "-s", "--", *command_args],
            input=script_text.encode("utf-8"),
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
