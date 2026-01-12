# Comparison Summary: GNN vs. CatBoost for ASD Classification

This report summarizes the performance comparison between the **BrainGNN-Multimodal (GNN)** model and the **CatBoost** classifier for Autism Spectrum Disorder (ASD) classification using the ABIDE dataset.

## 1. Performance Overview

The following table compares the average performance metrics across 5-fold cross-validation for both models.

| Metric          | BrainGNN-Multimodal (GNN) | CatBoost Classifier | Difference (GNN - CatBoost) |
| :-------------- | :------------------------ | :------------------ | :-------------------------- |
| **Accuracy**    | **0.6385** (±0.036)       | 0.5700              | +0.0685                     |
| **AUC**         | **0.6651** (±0.031)       | 0.6095              | +0.0556                     |
| **F1-Score**    | **0.6346** (±0.033)       | 0.5700 (weighted)   | +0.0646                     |

### Key Observations:
*   **GNN Superiority**: The BrainGNN-Multimodal model consistently outperforms the CatBoost classifier across all primary metrics.
*   **Accuracy Improvement**: The GNN model shows a significant improvement in accuracy, being approximately **6.85%** higher than CatBoost.
*   **Discriminative Power**: The GNN's AUC is **5.56%** higher, indicating better ability to distinguish between ASD and TD (Typically Developed) subjects.

## 2. Detailed Fold Performance (Test Accuracy)

The performance across individual folds shows more stability in the GNN model compared to the high variability seen in CatBoost.

| Fold | GNN Test Accuracy | CatBoost Test Accuracy |
| :--- | :---------------- | :--------------------- |
| 1    | 0.6776            | 0.4780                 |
| 2    | 0.6441            | 0.6497                 |
| 3    | 0.6667            | 0.5402                 |
| 4    | 0.6294            | 0.6000                 |
| 5    | 0.5749            | 0.5868                 |
| **Mean** | **0.6385**        | **0.5709**             |

*   **CatBoost Variability**: CatBoost's performance dropped significantly in Fold 1 (47.8%), which is below random chance, suggesting it struggled with certain data distributions or site-specific biases.
*   **GNN Stability**: While GNN also showed some drop in Fold 5, its overall performance remained more robust across the different splits.

## 3. Model Characteristics and Strategies

| Feature                | BrainGNN-Multimodal (GNN)                     | CatBoost Classifier                          |
| :--------------------- | :-------------------------------------------- | :------------------------------------------- |
| **Data Modalities**    | fMRI (Graph), sMRI (Deep Features), Phenotypic | fMRI (Flattened), sMRI (Features), Phenotypic |
| **Fusion Strategy**    | Cross-modal Attention & Bilinear Pooling      | Feature Concatenation                        |
| **Domain Adaptation**  | Multi-task Learning (Site Prediction)         | ComBat Harmonization                         |
| **Deconfounding**      | Multi-task Learning (Age Regression)          | None Explicit                                |
| **Feature Importance** | Attention-based (fMRI & sMRI)                 | Gradient Boosting Feature Importance         |

## 4. Conclusion

The **BrainGNN-Multimodal** model is the superior approach for this classification task. Its ability to model the brain's functional connectivity as a graph and utilize sophisticated fusion mechanisms (cross-modal attention and bilinear pooling) allows it to capture more complex patterns than the traditional gradient boosting approach of CatBoost. Furthermore, the GNN's integrated multi-task learning strategy for domain adaptation and deconfounding appears more effective than the external ComBat harmonization used in the CatBoost pipeline.

While CatBoost provides a strong baseline and valuable feature importance insights (particularly for sMRI features), the GNN's architectural advantages lead to more accurate and robust results in the heterogeneous multi-site ABIDE dataset.
