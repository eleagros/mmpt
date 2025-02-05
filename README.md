# Processing Mueller matrices

[![Twitter](https://img.shields.io/twitter/follow/horao_eu?style=flat)](https://twitter.com/horao_eu)

Automatic Mueller Matrix processing and plotting of the different polarimetric parameters.

## How to install

Download the repository using:
```sh
git clone https://github.com/RomGr/processingMM.git
```
Install the processingmm package by using:
```sh
cd processingmm
pip install -e .
```
If you want to use denoising (Work from [S. Moriconi](https://github.com/stefanomoriconi)), please contact me to get the models.

## How to use
The package is for the moment developped to be used using Jupyter.

**Install** Jupyter if not installed already:
```sh
pip install jupyter
```

**Launch** Jupyter notebook:
```sh
jupyter notebook
```

And **open** the notebook `process_MM_batch.ipynb`.

-----

You need to set several parameters:

**`directories`**: the path to the folder containing the measurement folders.

**`calib_directory`**: the path to the folder containing the calibration folders.

**`parameter_set`**: the set of parameters to be used for the line visualisation (file accessible at `./src/processingmm/data/parameters_visualisations.json`).

**`run_all`**: boolean indicating if the pipeline should be ran on all the folders (even the ones already processed).

**`wavelengths`**: set the wavelengths to be processed

- 'all': processes all the available wavelenght
- [xxx, yyy]: processes only the wavelenghts 'xxx' and 'yyy'

**`processing_mode`**:

- 'no_viz': processes only the MM - no visualization at all. useful for fast computation
- 'default': processes the MM and plots the polarimetric parameters maps (i.e. depolarization, azimuth, retardance, diattenuation, azimuth local variability)
- 'full': do like default, and additionally plot the MM components, as well as the line

**`save_pdf_figs`**: define if pdf figures should be saved (takes a lot of time) - no impact when processing_mode is set to no_viz

NB: processing time without PDDN
- 'no_viz': 0.71s
- 'default', save_pdf_figs False: 2.25s
- 'default', save_pdf_figs True: 3.60s
- 'full', save_pdf_figs False: 3.95s
- 'full', save_pdf_figs True: 8.07s

-----

And then run the cell containing the function `batch_processing.batch_process_master`.

-----

The visualization of the lines can be created using the function `batch_processing.batch_visualization`.

-----

The function `align_wavelengths.align_wavelenghts` allows to align the measurements captured at different wavelengths, which could cause issue when using masks obtained at 550nm as the images are slightly shifted 

-----

The function `add_calibration.add_calibration` add the calibration matrices (`A.cod` and `W.cod`) to the folder for processing (required by Chris)