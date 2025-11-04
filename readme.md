# DS Capstone Project: Signal Sigma

## Table of Contents

- [Disclaimer](#disclaimer)
- [Table of Contents](#table-of-contents)
- [Synopsis](#synopsis)
- [Installation](#installation)
- [Usage](#usage)
- [Colophon](#colophon)

## <a name="disclaimer"></a>Disclaimer<small><sup>[↩](#table-of-contents)</sup></small>

This project is intended solely for research, educational, and prototyping purposes. It does not constitute financial advice and should not be used for real-world trading or investment decisions without independent verification and consultation with a qualified financial advisor.

## <a name="synopsis"></a>Synopsis<small><sup>[↩](#table-of-contents)</sup></small>

**Signal Sigma** is a financial forecasting pipeline that ingests historical stock and macroeconomic data from sources like Yahoo Finance and FRED, and engineers a robust set of technical and composite macro features. A dedicated feature selection module streamlines the most relevant predictors for modelling.

The pipeline leverages a **Temporal Fusion Transformer** to deliver both point and probabilistic forecasts, complete with uncertainty bounds. An integrated Streamlit dashboard and comprehensive visualizations allow users to interactively explore predictions and key evaluation metrics.

## <a name="installation"></a>Installation<small><sup>[↩](#table-of-contents)</sup></small>

### Requirements

- Python 3.11.3
- pyenv

### Setup

1. Navigate to a working directory of your choice, then clone the repository and enter it:

   ``` shell
   git clone https://github.com/julialoeschel/capstone-SignalSigma.git &&
       cd capstone-SignalSigma
   ```

2. Choose a setup option based on your operating system and intended use:

   - `make basic-unix` / `make basic-win`: for general use or exploration (core dependencies only).
   - `make dev-unix` / `make dev-win`: for contributors (includes development tools like linters and pre-commit hooks).

   If you prefer to run the commands manually yourself or want to inspect what each `make` target does first, use the `-n` flag for a dry run. This prints the commands without executing them:

   ``` shell
   make -n <target>
   ```

3. Activate the virtual environment:

   - On macOS/Linux, run:

     ```shell
     source .venv/bin/activate
     ```

   - On Windows (PowerShell), run:

     ``` powershell
     .\.venv\Scripts\Activate.ps1
     ```

### Clean-Up

Besides performing a complete purge by manually deleting the project directory, you can use the following `make` targets to selectively clean the directory:

- `make clear-unix` / `make clear-win`: Removes build artifacts from the `data/`, `logs/`, `plots/` directories.
- `make reset-unix` / `make reset-win`: Removes, in addition to the previous option, also the virtual environment.

For a dry run preview, use the `-n` flag.

## <a name="usage"></a>Usage<small><sup>[↩](#table-of-contents)</sup></small>

``` shell
sisi [SUBCOMMAND|start] [ARGS] ...
```

If you prefer a more verbose binary identifier, you can also use `signal-sigma` instead of `sisi`.

### Subcommands

- `start`: Start Streamlit frontend.
- `forecast`: Run only the core functionality, write the inferred forecast to csv files on disk.
- `help`: Show help.

### Examples

``` shell
# Creates and writes forecast data frames for NVDA
# with re-run of pipeline.
sisi-forecast -r -t NVDA

# Creates and writes forecast data frames for a variety of stocks.
for ticker in AAPL AMZN GOOGL META MSFT NVDA TSLA; do
  sisi-forecast -t "${ticker}"
done

# Starts Streamlit frontend.
sisi
```

## Colophon

**Authors:** [julialoeschel](https://github.com/julialoeschel), [payamoghtanem](https://github.com/payamoghtanem), [Benas67](https://github.com/Benas67), [kvn-dtrx](https://github.com/kvn-dtrx)

**Template:** This repository was created from the [Neue Fische DS Capstone Project Template](https://github.com/neuefische/ds-capstone-project-template).

**License:** [MIT License](license.txt)
