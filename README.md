# Macrophages restrict tumor immune infiltration by controlling local collagen topography

This is the official codebase to the Random Forest classification part for the **Macrophages restrict tumor immune infiltration by controlling local collagen topography** paper.

## Context

The extracellular matrix (ECM), which constitutes the architecture of tissues, is drastically modified during tumorigenesis. These modifications support tumor growth and facilitate metastasis but also limits the immune response against the tumor, in particular by trapping immune cells in the capsula of collagen fibers surrounding tumor islets.  

As such, we use machine learning methods to better understand the extracellular matrix topology and immune cell behavior depending on the presence (WT) or depletion (DTR) of macrophage.

## Installation

Install the required packaged with conda using the environment.yml file running the following command:  

```
conda env create -f requirements.yml -n ecm
```

## How to use TumorMEC

Activate the working environment (after installation):  

```
conda activate ecm
```

Then you can either run the different jupyter notebooks `main.ipynb`, `main_base_on_timepoint.ipynb`, `main_time_on_other-time.ipynb`, `main_alltime_on_alltime.ipynb`.  
  
Or run the scripts in the `scripts/` directory example:  
```
python scripts/main_wt_cd3.py
```

## Contacts

Zoe FUSILIER (zoe.fusilier@curie.fr/zoe.fusilier@unige.ch)  
Hélène MOREAU (helene.moreau@curie.fr)
