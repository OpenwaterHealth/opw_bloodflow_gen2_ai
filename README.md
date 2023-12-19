## Gen2 Bloodflow Analysis

**Blood Flow Waveform Analysis and Classification Scripts for the Openwater Gen2 Bloodflow Device**

This repository houses Python scripts for analyzing and classifying blood flow waveforms, acquired with the Openwater Gen2 bloodflow device, to detect Large Vessel Occlusions(LVOs). The device was validated through a [comparative study](https://www.medrxiv.org/content/10.1101/2023.10.11.23296612v1) with Transcranial Doppler (TCD) and used in a [clinical study](https://www.medrxiv.org/content/10.1101/2023.12.14.23299992v1) to distinguish between Large Vessel Occlusion(LVO) and non-LVO strokes. Refer to the [company wiki](https://wiki.openwater.health/index.php/Openwater_Wiki), [technology summary wiki](https://wiki.openwater.health/index.php/Openwater_Stroke_Diagnosis_Technology) and [analysis software wiki](https://wiki.openwater.health/index.php/Blood_Flow_Gen_2_LVO_Classification_and_Analysis) for further context.

**Requirements:**

* Python 3.9.6 (other versions may work)
* pip

**Getting Started:**

1. Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

2. **Modify the `biosppy` library:**

The script `tools/ppg.py` modifies the bandpass filter in `biosppy` to the correct range for the Gen2 device's sampling rate. Replace the original `ppg.py` file (location depends on your environment) with the patched version:

* Anaconda: `<ENVIRONMENT_PATH>/lib/python3.9/site-packages/biosppy/signals/ppg.py`
* Virtualenv: `<VIRTUALENV_PATH>/lib/python3.9/site-packages/biosppy/signals/ppg.py`

**Repository Components:**

* **Sample Scan:** A sample scan is available in `SampleData`. Visualize it using the script `PlotOnDevice.py`

* **Signal Processing and Feature Extraction:** Implemented in the base class `ReadGen2Data.py`. `RunFeatureAnalysis.py` uses these features for patient population analysis for LVO vs non-LVO stroke classification. Patient data is not available due to contractual obligations with our clinical partners.

* **Deep Learning for LVO classification:** Two versions of deep learning networks are included:

    * **Resnet 1D:** Based on the implementation available [here](https://github.com/hsd1503/resnet1d). Scripts `DeepLearning/BloodFlowExperiments64.py` and `DeepLearning/BloodFlowExperiments128.py` train and evaluate the network.
    * **Transformer 1D:** Published in our [clinical paper](https://www.medrxiv.org/content/10.1101/2023.12.14.23299992v1) and used in our [FDA breakthrough](https://wiki.openwater.health/index.php/Regulatory) designation request. Implemented in `DeepLearning/transformer1D.py`.

**Contribution Guide:**

For more information on how to contribute to the project, please refer to the [Contribution Guide](CONTRIBUTING.md).

**Investigational Use Only**

CAUTION - Investigational device. Limited by Federal (or United States) law to investigational use. opw_bloodflow_gen2_ai has *not* been evaluated by the FDA and is not designed for the treatment or diagnosis of any disease. It is provided AS-IS, with no warranties. User assumes all liability and responsibility for identifying and mitigating risks associated with using this software.

**Additional Notes:**

* Consider updating the `biosppy` library directly to accept the sampling rate as an input for optimal solution.
* Feel free to adapt and modify the scripts to your specific needs. Improvements to the code are always welcome!
