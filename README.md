# dSHERLOCK

This repository holds the code to analyze dSHERLOCK reactions.

## Code

Explanation of modules in Analysis:

| Module  | Use | Input data format | Output data format |
| ------------- | ------------- | ------------- | ------------- |
| Image  | extracting fluorescence over time for each partition from image files  | .tif (two-channel image series) | .csv (timeseries of intensity for each partition) |
| Timeseries  | extracting features from fluorescence over time | .csv (timeseries of intensity for each partition)  | .csv (features extracted after selected pseudo-end-timepoint) |
| Auxiliary | helper functions | N.A. | N.A. |

Modules in Figures contain helper functions for plotting.

Explanation of Jupyter notebooks:

| Notebook  | Use |
| ------------- | ------------- |
| Image_processing_Example_data.ipynb  | applying the Image and Timeseries pipelines to the example image file, giving timeseries and features as results  |
| Thresholding_Example_data.ipynb  | determining fraction of positive compartments through thresholding (most commonly maximum intensity feat_fq_delta_max) for simple positive vs. negative |
| Classification_Example_data.ipynb | applying a pre-trained classifier for allele fraction quantification |
| Classification.ipynb | training a classifier and subsequently using the classifier for allele fraction quantification. NOTE: using this notebook needs the full training dataset and the full admix dataset which are both very large and its presence is intended more as a read-only reference to understand the training pipeline |

Pipeline explanation:
First, the Image and Timeseries scripts need to be applied to newly recorded data. See jupyter notebook Image_processing_Example_data.ipynb for an example.
Subsequently, there are two options:
- thresholding on one extracted feature (most commonly maximum intensity feat_fq_delta_max) for simple positive vs. negative. See jupyter notebook Thresholding_Example_data.ipynb for an example.
- classification with pre-trained classifier for allele fraction quantification. See jupyter notebook Classification_Example_data.ipynb for an example.


## Data

The included data was used to generate the following figures:

| Name  | Figure | Subfigures | Comment |
| ------------- | ------------- | ------------- | ------------- |
| CAuris CalCurve  | 2  | a-e  |  |
| Simulated Swab Samples  | 2 | f-g  |  |
| Timecourse 15h  | 3 | c-d  |  |
| FKS1 CalCurve  | 3 | e-f  | split into Mutant (MT) and Wildtype (WT) |
| Training  | 3 | h-j  | partial reuse of data from "Timecourse 15h" and "FKS1 CalCurve" for training |
| Admix  | 4 | a-l  |  |

Example_data contains some example data folders selected from the above data for illustration purposes.

## Contributing

If you would like to contribute, please reach out to Anton Thieme <anton@thiemenet.de>.

## License

[MIT](https://choosealicense.com/licenses/mit/)
