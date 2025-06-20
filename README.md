# ORCA2 MLE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15705427.svg)](https://doi.org/10.5281/zenodo.15705427)


## Context and Motivation

Purpose of this experiment is to compute the vertical buoyancy fluxes (VBF) induced by different submesoscale Mixed Layer Eddies (MLE) parameterisation in the NEMO [ORCA2](https://sites.nemo-ocean.io/user-guide/cfgs.html#orca2-ice-pisces) reference config.
Internal and external computed fluxes are written in an output file with the NEMO output system (XIOS).

> _**Objective is also to showcase hybrid Ocean/ML with NEMO5**_


#### Variations
- **BBZ24** : Velocity streamfunctions computed by inverting VBF infered with pre-trained [Bodner, Balwada and Zanna (2024)]() CNN


## Experiments Requirements


### Compilation

- NEMO version : [v5.0.1](https://forge.nemo-ocean.eu/nemo/nemo/-/releases/5.0.1) patched with [morays](https://github.com/morays-community/Patches-NEMO/tree/main/NEMO_v5.0.0) and local `CONFIG/src` sources.

- Compilation Manager : none, use standard `makenemo` script


### Python

- Eophis version : [v1.0.1](https://github.com/meom-group/eophis/releases/tag/v1.0.1)
- **BBZ24** dependencies:
  ```bash
    cd ORCA2.BBZ24/INFERENCES/BBZ24_MLE/
    pip install -e .  
  ```

### Run

- Production Manager: none, use submission script `job.ksh` in `RUN`

- Input files: 
  ```bash
    wget "https://gws-access.jasmin.ac.uk/public/nemo/sette_inputs/r5.0.0/ORCA2_ICE_v5.0.0.tar.gz"
  ```

### Post-Process

- No post-process libraries, use direct NEMO outputs

- `ORCA2_MLD.nc` file in `POSTPROCESS` contains ORCA2 MLD results without MLE
  
- Plotting : Python script `plot_res.py` in `POSTPROCESS`
