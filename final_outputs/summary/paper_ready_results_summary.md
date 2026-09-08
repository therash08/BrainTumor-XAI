# TrustSeg-XAI — Final Results Summary

## Classification Performance

On the locked 1,000-image official test set, EfficientNet-B0 achieved the
highest classification accuracy (0.9960) and Macro-F1
(0.9965). TrustSeg-XAI achieved an accuracy of
0.9900 and Macro-F1 of 0.9906,
while ConvNeXt-Tiny achieved 0.9860 accuracy and
0.9872 Macro-F1.

Bootstrap 95% confidence intervals were
[0.9920, 0.9990]
for EfficientNet-B0 accuracy,
[0.9840, 0.9960]
for TrustSeg-XAI, and
[0.9780, 0.9930]
for ConvNeXt-Tiny.

Exact paired McNemar testing showed that EfficientNet-B0 outperformed
ConvNeXt-Tiny in paired correctness
(p=0.0129395), whereas the EfficientNet-B0 versus TrustSeg-XAI
comparison was not statistically significant
(p=0.145996). The ConvNeXt-Tiny versus TrustSeg-XAI
difference was also not significant
(p=0.454498).

## Segmentation Performance

On the 860 tumor-positive MRI with expert segmentation masks, TrustSeg-XAI
achieved a mean Dice score of 0.8392, compared with
0.8321 for U-Net ResNet34 and 0.8353 for
U-Net++ ResNet34. TrustSeg-XAI therefore had the highest local point estimate
for Dice among the three segmentation models. Its bootstrap 95% confidence
interval for mean Dice was
[0.8280, 0.8495].
No pairwise statistical claim is made for the segmentation-model differences.

## Mask Dependency

Changing the explicit ROI guidance produced no hard classification changes
among any of the 1,000 test MRI. Predicted, target, no-mask and corrupted-mask
guidance all retained an accuracy of 0.9900 and Macro-F1 of 0.9906.

Removing guidance changed mean true-class probability by only
-0.000361, while spatially corrupting
guidance changed it by -0.000369.
Although these small probability shifts were statistically detectable, their
magnitude was very small. The results therefore indicate negligible
hard-decision dependence and only very weak soft dependence on the explicit
ROI-guidance pathway.

## Grad-CAM++ Anatomical Localization

Grad-CAM++ was evaluated on the permanently fixed, jointly stratified
150-image tumor subset. The mean XAI Dice was
0.3418, mean XAI IoU was
0.2361, Pointing Game accuracy was
0.3733, and the mean fraction of saliency
inside the expert tumor region was
0.1950. There were
17 degenerate CAMs.

The bootstrap 95% confidence interval for XAI IoU was
[0.2044, 0.2685].
These results indicate partial but incomplete anatomical localization rather
than consistently strong localization.

## Causal Tumor Faithfulness

Direct tumor occlusion produced a mean target-class probability drop of
0.2765, whereas matched-background
occlusion produced a mean drop of only
0.0002. The mean causal
faithfulness margin was therefore
0.2763, with a bootstrap 95%
confidence interval of
[0.2051, 0.3499].

Tumor removal had a greater probability effect than matched-background removal
in 62.0% of cases.
Tumor occlusion changed the target decision in
28.0% of cases, whereas the matched
background control changed it in
0.0%. The one-sided paired Wilcoxon
test was significant (p=3.29625e-08).

However, the median faithfulness margin was
0.000001, indicating substantial
case-level heterogeneity. The evidence therefore supports strong average
causal tumor dependence, but not uniformly strong dependence for every MRI.

## No-Tumor Safety

On all 140 official no-tumor MRI, TrustSeg-XAI achieved
100.0% classification accuracy.
The mean total tumor-class probability was only
0.00000415.

At the permanently fixed segmentation threshold of 0.5, the mean and maximum
false-tumor area ratios were both
0.00000000 and
0.00000000, respectively.
The empty-mask rate was
100.0%, and no false-tumor
segmentation detections occurred in this cohort.

These findings indicate excellent no-tumor safety on the evaluated official
test cohort, without implying universal clinical safety.

## Calibration

EfficientNet-B0 was the best-calibrated classifier, with ECE
0.003536 and multiclass Brier score
0.007649. TrustSeg-XAI had ECE
0.009990 and Brier score
0.019438, while ConvNeXt-Tiny had ECE
0.011382 and Brier score
0.024853.

All three classifiers were overconfident on average, with TrustSeg-XAI having
mean confidence 0.999375 despite accuracy
0.990000.

## Integrated Trustworthiness Interpretation

The final experiments separate several dimensions of trustworthiness.

TrustSeg-XAI achieved strong diagnostic classification and the highest local
segmentation Dice point estimate among the evaluated segmentation models.
However, classification decisions showed negligible dependence on the explicit
ROI-guidance mask, indicating that the global ConvNeXt pathway dominated the
hard classification decision.

At the same time, directly removing expert-defined tumor content from the MRI
produced a large average probability reduction relative to area-matched
background removal. Thus, weak explicit mask dependence does not imply weak
dependence on tumor image content.

Grad-CAM++ provided only partial anatomical localization, whereas the causal
occlusion experiment demonstrated substantially stronger average tumor
dependence. These results show that post-hoc saliency localization and causal
faithfulness capture distinct properties and should not be treated as
interchangeable evidence of model trustworthiness.
