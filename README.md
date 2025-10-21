# AI-Based Detection and Quantification of Pulmonary Nodules (Advanced)

**Project:** AI-Based Detection and Quantification of Pulmonary Nodules in HRCT Thorax Studies
**Course:** Healthcare Data Analytics (BCSE335L)
**Team:** Parth Suri (22BDS0116), Kushal Sharma (22BCE2561), Akanksh G. Prabhu (22BCE0122)

This repository contains an advanced, research-grade pipeline for pulmonary nodule detection & quantification.
It supports:
- DICOM -> NIfTI preprocessing (SimpleITK)
- MONAI-based 3D U-Net segmentation
- 3D ResNet patch classifier for FP reduction (two-stage detector)
- Hard-negative mining, augmentation, mixed-precision training
- Inference, candidate extraction, FROC evaluation, volumetric quantification

**Quick demo:** the `src/synth_demo.py` script generates synthetic volumes so you can run a quick end-to-end test before using real data.
See `requirements.txt` for required packages.
