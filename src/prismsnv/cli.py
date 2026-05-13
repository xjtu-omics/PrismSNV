import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

COMMANDS = ("bam2vcf", "snv2barcode", "pre_train", "snv_effect")


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
                  prismsnv snv_effect -y <train_config.yaml>

            Typical workflow:
              1. prismsnv bam2vcf ...
              2. prismsnv snv2barcode <snv2barcode_config.yaml>
              3. prismsnv pre_train -y <train_config.yaml>
              4. prismsnv snv_effect -y <train_config.yaml>

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

    if command == "pre_train":
        from .train.pre_train import main as pre_train_main

        pre_train_main(command_args)
    elif command == "snv_effect":
        from .train.snv_effect import main as snv_effect_main

        snv_effect_main(command_args)
    elif command == "snv2barcode":
        if len(command_args) != 1 or command_args[0] in {"-h", "--help"}:
            parser.exit(
                0 if command_args and command_args[0] in {"-h", "--help"} else 2,
                "Usage: prismsnv snv2barcode <path_to_config.yaml>\n",
            )

        from .preprocess.snv2barcode import main as snv2barcode_main

        snv2barcode_main(command_args[0])
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
