# MPhil Data Intensive Science Image Analysis

This repository contains all the code used for generating figures and results in the report.

## Installation
### Downloading the repository

To download the repository, simply clone it from the GitLab page:

Clone with SSH:
```
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/a8_img_assessment/dw661.git
```

Clone with HTTPS:
```
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/a8_img_assessment/dw661.git
```

## Usage
### Running scripts
All results and figures included in the report can be reproduced by running relevant scripts. Specifically:
- `Q1-CT.py` produces figures used for segmentation task of CT scans in Question 1.
- `Q1-flowers.py` produces figures used for segmentation task of purple tulips in Question 1.
- `Q1-coins.py` produces figures used for segmentation task of coins in Question 1.
- `Q2-1.py` produces results and figures used for part 1 of Question 2.
- `Q2-2.py` produces results and figures used for part 2 of Question 2.
- `Q2-3.py` produces results and figures used for part 3 of Question 2.
- `Q3-1.py` contains script used for Gradient Descent used in part 1 of Question 3.
- `Q3-2.py` contains script used for training the Learned Gradient Desceent network, as well as script used for evaluation of other benchmarking methods, such as FBP and TV. The completed implementations of the networks are migrated to the file `LGD_models.py`; this file contains the completed implementations of `prox_net` and `LGD_net`. The completed training scripts for the network is included in `Q3-2.py`.

To run scripts, simply run the desired file in command line:

```{Python}
python src/Q1-coins.py # This will generate all figures used in the coins segmentation in Question 1
```

## Docker Instructions
All the packages used and their versions were included in the file `environments.yml`. This file can be used to replicate the Conda environment used during the development of this repository.

To run the Python scripts inside Docker, first build the image

```
docker build -t ia .
```
This would generate an image called `ia`. To deploy and run the container, run the following command:

```
docker run --rm -ti ia
```
This would start the process inside the container.

## Hardware Specifications
All scripts should be runnable on most machines in less than 1 minutes; `Q3-2.py` may take slightly longer as a network is trained.

## Environment
The exact environment used for all the code development is detailed in the file `environment.yml`.

## Documentation
All functions in the repository are annotated with documentations and comments; to explicitly generate documentation, run `doxygen` inside of `docs`; this should populate `docs` with HTML and Latex documentations.

## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgement of Generative AI Tools
During the completion of coursework, generative tools such as ChatGPT and CoPilot were used supportively and minimally. All code involving any algorithms or calculations were entirely produced by myself; Copilot was only partially used for Docstring and plotting, and ChatGPT was only used for latex syntax queries. Examples of prompts include:

"How to align plots properly in LaTex document?"