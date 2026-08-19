
# ECG-Quantification

## Beyond Single-Beat Classification: Quantifying Arrhythmia in Long-Term ECG via Prevalence Estimation

### Abstract
Long-term electrocardiogram (ECG) monitoring is essential for determining the arrhythmic burden, a critical clinical metric for diagnosing cardiovascular conditions. Traditionally, this burden is estimated using a Classify-and-Count (CC) approach, which labels individual heartbeats and aggregates results by counting predictions for each label. However, even state-of-the-art Deep Learning classifiers exhibit systematic biases that accumulate over long-term recordings, leading to significant diagnostic inaccuracies. This paper investigates the application of quantification techniques to estimate arrhythmia prevalence in long-term ECG signals from the MIT-BIH Arrhythmia Database. We compare several base classifiers paired with quantification algorithms against a high-performance Deep Learning baseline, LITETime, using the standard CC method. Our results demonstrate a quantification paradox: while the LITETime achieves superior beat-by-beat accuracy, simpler classifiers equipped with quantification adjustment layers, particularly the Expectation-Maximization Quantifier (EMQ), significantly reduce the Mean Absolute Error (MAE) in prevalence estimation. By correcting the systematic bias caused by Prior Probability Shifts, our framework provides a more reliable diagnostic tool for long-term monitoring and wearable cardiac devices.

---

This repository contain all files needed to run the experiments for the CBMS ECG Quantification paper.

In order to make it more easier, all experiments are stored in the jupyter notebook format, inside of notebooks directory. All notebooks and files must be in order to execute (01_, 02_, ...). All the data is fetched from the PhysioNet python library, there is no need to manually download them.

For future organization, all files will be converted to python scripts, and become a lib format.
