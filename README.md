# Rubin Observatory SIT-Com Verification and Validation Notebook Repository
This repository is to store and organize notebooks and any associated methods which have a Zephyr JIRA Test Case counterpart.  
The reason for creation of this repo is to make it easier to execute each Test Case in a systematic fashion.  

To keep the size of the repository small and therefore faster to clone/manage, it is recommended to clear the notebooks of rendered content before committing via git.
It is also recommended that notebooks and/or associated methods that may need to be used by others contain a minimal amount of documentation and/or comments to provide context and/or instructions.


## Notebooks

User notebooks should be stored in the notebooks directory.
The folder structure inside the `notebooks` directory should approximately reflect the organization in Zephyr JIRA.
The folder/file organization is not set on stone, but one should consider how easy is to find the Notebook counterpart of a Test Case when placing it somewhere.  

Each notebook's name should start with the test case code, a dash, and short name that represent the test case.  
This will help quick assesment of each notebook's function while keeping the information required to find the actual Test Case.
Do not use spaces in the filename.  
Instead, replace spaces with underlines (`_`). 

For example, you can find the [LVV-T2229 (v2.0) Closed Loop ComCam Image Ingestion and Application of Correction] notebook counterpart in `notebooks/proj_sys_eng/SIT-COM_Integration/LVV-T2229-Closed_Loop.ipynb`.  

[LVV-T2229 (v2.0) Closed Loop ComCam Image Ingestion and Application of Correction]: https://jira.lsstcorp.org/secure/Tests.jspa#/testCase/LVV-T2229

A sub-folder should have at least five notebooks within it to justify its existence.  
Otherwise, simply add the notebooks in the current folder.  
This avoid having multiple sub-folders with very few files within them and make it easier to find the files.


## Methods

User methods developed to support notebooks should be stored in the python directory.
It is strongly recommended to follow Rubin development formats/practices to standardize behavior and minimize the overhead when sharing/running each others code.
This repo is eups compatible.
If a user wishes to develop their own support methods, this repo must be setup prior to importing them.

One way to setup this repo is to add the following to the `~/notebooks/.user_setups` file:

    setup -j notebooks_vandv -r ~/notebooks/lsst-sitcom/notebooks_vandv
    
You can replace `~/notebooks/lsst-sitcom/notebooks_vandv` with the directory where this file is located.


## Tests

Unit tests should be stored in the tests directory.
