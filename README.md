# Mueller Matrix Processing Toolbox (MMP)

MMP is a toolbox allowing to process automatically Mueller Matrices and plotting the different polarimetric parameters.

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

### You then need to follow the installation instructions from [libmpMuelMat](https://github.com/stefanomoriconi/libmpMuelMat)

It is rather straightforward in Unix based OS, but can be more complicated in Windows.

    [Requirements] 
    
    C/C++ Compiler + Parallel and Multi-Processing libraries: 
    	
    	* Linux - Ubuntu 20.04.2 LTS
    	- GCC 9.4 or later (default) -- https://gcc.gnu.org/ 
    	- OpenMP (included in GCC) -- https://www.openmp.org/ 

    	* Mac - MacOS Catalina 10.15.7
    	- Homebrew 3.5.7   [ from terminal: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"]
    	- GCC 9.4 or later [ from terminal: brew install gcc ]
    	- OpenMP (included in GCC) -- https://www.openmp.org/

    	* Windows - Win10 or later
    	- MSYS2 -- https://www.msys2.org/ 
    	- GCC 9.4 or later -- install from MSYS2 (see below)
    	- OpenMP (included in GCC) -- https://www.openmp.org/

	*** WINDOWS ***
	1) Download MSYS2:
	https://github.com/msys2/msys2-installer/releases/download/2022-09-04/msys2-x86_64-20220904.exe

	2) Follow the instructions to install MSYS2 on:
	https://www.msys2.org/   -- click NEXT until complete
	
	3) Launch the MINWG64 environment:
	Start Menu >> Programs >> MSYS2 >> MSYS2 MINGW64
	
	4) Install GCC from the MINGW64 environment [terminal]:
	$ pacman -S mingw-w64-x86_64-gcc
	
	5) Verify GCC has been succesfully installed:
	$ gcc --version
	
		it should display '12.2.0' or later

    
    [Installation]

	* Compiling the C source code into the shared library 'libmpMuelMat.so'

		open (with a text editor) and check first instructions (commented lines) in ./src/processingmm/C-libs/compileMeFirst.sh

		[LINUX, MAC] after possible adjustments run in the terminal:

		$ cd ./src/processingmm/C-libs/C-libs
		$ bash compileMeFirst.sh

		[WINDOWS] after editing the *adjustments* (e.g. using notepad) launch the MINGW64 environment and run in the terminal:

		$ cd ./src/processingmm/C-libs/C-libs
		$ bash compileMeFirst.sh 
		
        [WINDOWS] once succesfully compiled, *add* the MINGW64 shared library folder to the PATHS of the System:

        Start >> Edit the System Environmental Variables >> Environment Variables... >> System Variables >> Path >> New: MSYS2_MINGW64_SHARED_LIBS_PATH

        with MSYS2_MINGW64_SHARED_LIBS_PATH = 'C:\msys64\mingw64\bin'  (by default)
 

    [Configuration]

	[WINDOWS ONLY]: edit (notepad) the python library '.src/processingmm/libmpMuelMat.py' lines: (94,95) to correctly reference the compiled shared library (.dll)
	

	* Loading the library in python (after launching e.g. ipython):

		>> import libmpMuelMat

	* Displaying the library dipendencies:

		>> libmpMuelMat.list_Dependencies()
		
		-- Please adjust the paths and dependencies to make it consistent, prior to using the tools.

	* Testing C-compiled source code correctly linking to multi-processing libraries (OpenMP)

		>> libmpMuelMat.test_OpenMP()

		-- If the parallel-computinf and multi-processing libraries are not correctly linked,
		   the library will not be optimised for performance.


You can verify the installation without (and with) denoising by running the following command:
```sh
python ./tests/test_processingMM.py
```

```sh
python ./tests/test_processingMM_PDDN.py
```
The output, if the installation is successful should be something like:
```sh
Ran 3 tests in 5.000s
OK
```

If everything works out, you should be good to go! 

## How to use

### 1. Script
The script **`process_mueller_matrices.py`** can be used to process the Mueller Matrices, the command is the following:

```sh
python process_mueller_matrices.py [-h] --directory DIRECTORY --calib CALIB [--PDDN_mode {no,pddn,both}] [--wavelengths WAVELENGTHS [WAVELENGTHS ...]]
                                   [--processing_mode {no_viz,default,full}] [--save_pdf_figs] [--run_all] [--align_wls]
```
- with *DIRECTORY* being the folder containing the measurements
- with *CALIB* being the folder containig the calibration folders

Run 
```sh
python process_mueller_matrices.py -h
```
for more information about the different parameters


### 2. Using a notebook
**Install** Jupyter if not installed already:
```sh
pip install jupyter
```

**Launch** Jupyter notebook:
```sh
jupyter notebook
```

And **open** the notebook **`process_MM_batch.ipynb`**.

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

