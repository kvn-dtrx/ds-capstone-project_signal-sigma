#!/usr/bin/env python3

# ---
# description: Entry point for the execution of the cli utility.
# ---

# ---

import os
import subprocess
import argparse

# TODO: Replace this with main as soon as possible
import signal_sigma.core.__main__ as sisi
import signal_sigma.config.cfg as cfg


def run_streamlit(_args) -> None:
    streamlit_path = os.path.join(
        cfg.SRC_PATH,
        "signal_sigma",
        "cli",
        "app.py",
    )
    try:
        print("🚀 Launching the Streamlit app...")
        subprocess.run(["streamlit", "run", streamlit_path], check=True)
    except KeyboardInterrupt:
        print("\n⛔ Streamlit app gracefully stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Streamlit encountered an error: {e}")


def run_forecast(_args) -> None:
    sisi.main()


def main() -> None:
    parser = argparse.ArgumentParser(prog="sisi")

    subparsers = parser.add_subparsers(dest="command")

    parser_ui = subparsers.add_parser(
        "ui",
        help="Run Streamlit frontend for Signal Sigma",
    )
    parser_ui.set_defaults(func=run_streamlit)

    parser_forecast = subparsers.add_parser(
        "forecast",
        help="Forecast with Signal Sigma",
    )
    parser_forecast.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to forecast",
    )
    # TODO: Add more arguments
    parser_forecast.set_defaults(func=run_forecast)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        args = parser.parse_args(["ui"])

    args.func(args)


if __name__ == "__main__":
    main()
