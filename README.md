# Knowledge XAIeve: A framework for discovering knowledge in panoramic radiographs through task-specific explainable AI and sample perturbation techniques---A Case Study of the Thai Population Aged 7–25

[Natthanich Hirunchavarod](https://github.com/nattntn), [Natnicha Sributsayakarn](https://www.phyathai.com/en/pyt2/doctor/d-d-s-natnicha-sributsayakarn), [Suchaya Pornprasertsuk-Damrongsri](https://murex.mahidol.ac.th/en/persons/suchaya-pornprasertsuk-damrongsri/), [Varangkanar Jirarattanasopha](https://dt.mahidol.ac.th/language/en/menu-personnel/varangkanar-jirarattanasopha-en/), [Thanapong Intharah](https://vi-lab-th.github.io)

> Abstract: We present Knowledge XAIeve, a framework for  discovering knowledge from convolutional neural  network (CNN) models trained on panoramic  radiographs, translating a model's internal  reasoning into explicit, testable hypotheses. 
The framework operates through five stages: (1) preparing the dataset, (2) training a  task-specific CNN model, (3) applying global  explainability to localize influential semantic  regions, (4) performing systematic perturbation  to quantify their influence, and (5) statistically  validating the resulting patterns using linear mixed-effects models.
We demonstrated its application using 5,132 panoramic radiographs from 2,778 Thai patients aged 7-25 years across two scenarios: sex classification (accuracy 87.38\%, sensitivity 87.61\%, specificity 87.16\%) and age estimation (RMSE 1.96 years). 
The framework reaffirmed the lower third molar as the primary predictor for age estimation, revealing a nonlinear, sex-dependent relationship between the crown--root ratio and predicted age ($p$ < 0.001), and reaffirmed the upper canine while discovering upper third molar as key predictors for male and female sex classification, respectively ($p$ < 0.001). 
Follow-up validation confirmed that sexual dimorphism in the upper third molar is driven primarily by root rather than crown dimensions.
These findings demonstrate the potential of Knowledge XAIeve for generating novel, biologically grounded hypotheses from CNN models trained on panoramic radiographs.

## Method Overview
