# ORCA2 MLE

[![DOI](https://zenodo.org/badge/763681074.svg)](https://doi.org/10.5281/zenodo.13851909)

## Context and Motivation

Purpose of this experiment is to compute the vertical buoyancy fluxes (VBF) induced by different submesoscale Mixed Layer Eddies (MLE) parameterisation in the NEMO [ORCA2](https://sites.nemo-ocean.io/user-guide/cfgs.html#orca2-ice-pisces) reference config.
Internal and external computed fluxes are written in an output file with the NEMO output system (XIOS).

#### Variations
- **BBZ24** : Velocity streamfunctions computed by inverting VBF infered with pre-trained [Bodner, Balwada and Zanna (2024)]() CNN


<img width="695" alt="MLE_EXP" src="https://github.com/morays-community/NEMO-MLE_Fluxes/assets/138531178/084171b2-7f5d-407b-ad6c-92551f3bbcb2">

## Experiments Requirements


### Compilation

- NEMO version : [v5.0.1](https://forge.nemo-ocean.eu/nemo/nemo/-/releases/5.0.1) patched with [morays](https://github.com/morays-community/Patches-NEMO/tree/main/NEMO_v5.0.0) and local `CONFIG/src` sources.

- Compilation Manager : none, use standard `makenemo` script


### Python

- Eophis version : [v1.0.1](https://github.com/meom-group/eophis/releases/tag/v1.0.1)
- **BBZ24** dependencies:
  ```bash
    git submodule update --init --recursive
    cd eORCA025_MLE.BBZ24/INFERENCES/NEMO_MLE
    pip install -e .  
  ```

### Run

- Production Manager: none, use submission script `job.ksh` in `RUN`

### Post-Process

- No post-process libraries
  
- Plotting : Python script `plot_res.py` in `POSTPROCESS`
