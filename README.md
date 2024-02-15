# Rubin Observatory SIT-Com Verification and Validation Notebook Repository
This repository is to store and organize notebooks and any associated methods which have a Zephyr JIRA Test Case counterpart.
The reason for creation of this repo is to make it easier to execute each Test Case in a systematic fashion.

To keep the size of the repository small and therefore faster to clone/manage, it is recommended to clear the notebooks of rendered content before committing via git.
It is also recommended that notebooks and/or associated methods that may need to be used by others contain a minimal amount of documentation and/or comments to provide context and/or instructions.

Some notebooks will require you to clone some extra repositories to your Nublado home folder.
Since every user has a different setup, the paths might be slightly different.
It is recommended to have all the repositories cloned under `$HOME/notebooks`.

You might end up with many repositories and adding an extra folder with the name of the organization they belong might help to find them on GitHub later.
For example, this repository would be located in `$HOME/notebooks/lsst-sitcom/notebooks_vandv`.
However this is simply a recommendation and you are free to organize your folders as you please.


## Notebooks

User notebooks should be stored in the notebooks directory.
Please, remember to stop the Kernel and clean all the notebooks outputs before committing and pushing them to GitHub.
The folder structure inside the `notebooks` directory should approximately reflect the organization in Zephyr JIRA.
The folder/file organization is not set on stone, but one should consider how easy is to find the Notebook counterpart of a Test Case when placing it somewhere.

Each notebook's name should start with the test case code, a dash, and short name that represent the test case.
This will help quick assesment of each notebook's function while keeping the information required to find the actual Test Case.
Do not use spaces in the filename.
Instead, replace spaces with underlines (`_`).

For example, you can find the [LVV-T2229 (v2.0) Closed Loop ComCam Image Ingestion and Application of Correction] notebook counterpart in `notebooks/proj_sys_eng/SIT-COM_Integration/LVV-T2229-Closed_Loop.ipynb`.

A sub-folder should have at least five notebooks within it to justify its existence.
Otherwise, simply add the notebooks in the current folder.
This avoid having multiple sub-folders with very few files within them and make it easier to find the files.

Data-analysis notebooks should live in the same folder as the data-acquisition notebooks.
Data-analysis notebooks should have the following filename `LVV-TXXXX-analysis.ipynb`, where `LVV-TXXXX` is the associated test case.

[LVV-T2229 (v2.0) Closed Loop ComCam Image Ingestion and Application of Correction]: https://jira.lsstcorp.org/secure/Tests.jspa#/testCase/LVV-T2229

## Requirements

This notebooks require some extra repositories to be cloned locally so it can grab some constants and some look-up tables.
Here is a list of which repos are required to run this notebook:

* [lsst-ts/ts_criopy]
* [lsst-ts/ts_m1m3support]
* [lsst-ts/ts_config_mttcs]
* [lsst-ts/ts_xml]
* [lsst-ts/ts_utils]
* [lsst-sitcom/M2_FEA]
* [lsst-sitcom/summit_utils]

[lsst-ts/ts_criopy]: https://github.com/lsst-ts/ts_criopy
[lsst-ts/ts_m1m3support]: https://github.com/lsst-ts/ts_m1m3support
[lsst-ts/ts_config_mttcs]: https://github.com/lsst-ts/ts_config_mttcs
[lsst-ts/ts_xml]: https://github.com/lsst-ts/ts_xml
[lsst-ts/ts_utils]:https://github.com/lsst-ts/ts_utils
[lsst-sitcom/M2_FEA]: https://github.com/lsst-sitcom/M2_FEA
[lsst-sitcom/summit_utils]: https://github.com/lsst-sitcom/summit_utils

Since every user has a different setup, the paths might be slightly different.
It is recommended to have all the repositories cloned under `$HOME/notebooks`.
You might end up with many repositories and adding an extra folder with the name of the organization they belong might help to find them on GitHub later.
For example, this repository would be located in `$HOME/notebooks/lsst-sitcom/notebooks_vandv`.
The paths below consider this directory structure but, of course, you are free to organize your folders as you please.

In order to have the required repositories available, open a terminal and run the following commands:

```
git clone https://github.com/lsst-sitcom/notebooks_vandv $HOME/notebooks/lsst-sitcom/notebooks_vandv
git clone https://github.com/lsst-ts/ts_criopy $HOME/notebooks/lsst-ts/ts_criopy
git clone https://github.com/lsst-ts/ts_m1m3support.git $HOME/notebooks/lsst-ts/ts_m1m3support
git clone https://github.com/lsst-ts/ts_config_mttcs $HOME/notebooks/lsst-ts/ts_config_mttcs
git clone https://github.com/lsst-ts/ts_utils $HOME/notebooks/lsst-ts/ts_utils
git clone https://github.com/lsst-ts/ts_xml $HOME/notebooks/lsst-ts/ts_xml
git clone https://github.com/lsst-sitcom/M2_FEA $HOME/notebooks/lsst-sitcom/M2_FEA
git clone https://github.com/lsst-sitcom/summit_utils.git $HOME/notebooks/lsst-sitcom/summit_utils
```

If you use a different path for these repositories, make sure that you pass this path when running the associated functions.

And add these lines to your `$HOME/notebooks/.user_setups` file:

```
export LSST_DDS_DOMAIN_ID=0
setup -j notebooks_vandv -r $HOME/notebooks/lsst-sitcom/notebooks_vandv
setup -j ts_criopy -r $HOME/notebooks/lsst-ts/ts_criopy
setup -j ts_config_mttcs -r  $HOME/notebooks/lsst-ts/ts_config_mttcs
setup -j ts_utils -r  $HOME/notebooks/lsst-ts/ts_utils
setup -j M2_FEA -r $HOME/notebooks/lsst-sitcom/M2_FEA
setup -j summit_utils -r $HOME/notebooks/lsst-sitcom/summit_utils
```

You might want to work using a special branch for one or more repositories.
In that case, go to the repository root folder and checkout the branch you want to use.
For example:

```
cd $HOME/notebooks/lsst-sitcom/summit_utils
git checkout ${work_branch_name}
```

Finally, you will need to put M1M3 and M2 to use the mount for the look-up table calculations.
For M2, you can check the [M2 Summit Manual] page in Confluence.

[M2 Summit Manual]: https://confluence.lsstcorp.org/display/LTS/Use+of+M2+EUI+on+Summit


## Methods

User methods developed to support notebooks should be stored in the python directory.
It is strongly recommended to follow Rubin development formats/practices to standardize behavior and minimize the overhead when sharing/running each others code.
This repo is eups compatible.
If a user wishes to develop their own support methods, this repo must be setup prior to importing them.

One way to setup this repo is to add the following to the `~/notebooks/.user_setups` file:

    setup -j notebooks_vandv -r ~/notebooks/lsst-sitcom/notebooks_vandv

You can replace `~/notebooks/lsst-sitcom/notebooks_vandv` with the directory where this file is located.


## Developer Guide

We will try to adopt most of the practices/workflow adopted by the Telecope and Site and Data-Management teams.
Here are the links for both of them:

- https://tssw-developer.lsst.io/
- https://developer.lsst.io/

Here are a few quick points to keep an eye on:

1. We want to follow [TS Development Workflow] (JIRA Ticket, new branch `ticket/PROJ-????`, Pull Request).
2. When writing documentation or text in notebooks, try to use [Semantic Like Breaks] for clarity.
3. For code standards, let's use the [PEP-8] as a reference.
4. When writing new plots/tools, consider the [Rule of Three] to avoid duplication/repetition.

[TS Development Workflow]: https://tssw-developer.lsst.io/work_management/development_workflow.html#development-workflow
[Semantic Like Breaks]: https://sembr.org/
[PEP-8]: https://peps.python.org/pep-0008/
[Rule of Three]: https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming)

(more to come)

## Tests

Unit tests should be stored in the tests directory.
