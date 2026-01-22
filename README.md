# Student Readiness Clustering (Repo)

Repository: student readiness clustering pipelines and notebooks. Implements an end-to-end workflow for preprocessing survey data, performing agglomerative clustering, evaluating candidate clusterings with multiple metrics, ranking cluster choices by consensus, assigning held-out test rows to clusters, and exporting visual summaries.

---

## Quick links (important files & symbols)
- Notebook: [student-readiness-for-class-course/agglomerative_clustering.ipynb](student-readiness-for-class-course/agglomerative_clustering.ipynb)  
- Helper module and key functions: [`helper.load_data`](student-readiness-for-class-course/helper.py) · [`helper.to_midpoint`](student-readiness-for-class-course/helper.py) · [`helper.preprocess`](student-readiness-for-class-course/helper.py) · [`helper.save_trained_model`](student-readiness-for-class-course/helper.py) — see [student-readiness-for-class-course/helper.py](student-readiness-for-class-course/helper.py)  
- Data: [student-readiness-for-class-course/data/survey.csv](student-readiness-for-class-course/data/survey.csv), [student-readiness-for-class-course/data/test_data_labelled.csv](student-readiness-for-class-course/data/test_data_labelled.csv), [student-readiness-for-class-course/data/hdbscan_clustered.csv](student-readiness-for-class-course/data/hdbscan_clustered.csv), [student-readiness-for-class-course/data/survey_clustered_kmeans.csv](student-readiness-for-class-course/data/survey_clustered_kmeans.csv)  
- Tests: [student-readiness-for-class-course/tests/test_pipeline_smoke.py](student-readiness-for-class-course/tests/test_pipeline_smoke.py), [student-readiness-for-class-course/tests/conftest.py](student-readiness-for-class-course/tests/conftest.py)  
- Visual outputs: [student-readiness-for-class-course/visualisations/](student-readiness-for-class-course/visualisations/)

---

## Purpose
This repo demonstrates a reproducible pipeline to transform labelled student survey responses into actionable student segments via hierarchical (agglomerative) clustering. Goals:
- Reliable preprocessing of mixed-format survey answers (ranges, binary, categorical, missing values).
- Consistent train/test preprocessing to avoid leakage.
- Evaluate clustering candidates using multiple metrics and a consensus ranking scheme.
- Provide visual and CSV summaries to interpret cluster profiles.

---

## Contents & structure
- student-readiness-for-class-course/agglomerative_clustering.ipynb — orchestration: load → preprocess → cluster → evaluate → predict → visualize.  
- student-readiness-for-class-course/helper.py — core utilities:
  - [`helper.load_data`](student-readiness-for-class-course/helper.py): deterministic train/test split (68.75% / 31.25%).  
  - [`helper.to_midpoint`](student-readiness-for-class-course/helper.py): convert range strings like `3-5` → midpoint float.  
  - [`helper.preprocess`](student-readiness-for-class-course/helper.py): deterministic encoding (category maps), range handling, Yes/No mapping, fill/mask, and scaling with option to return scaler & encodings for reuse.  
  - [`helper.save_trained_model`](student-readiness-for-class-course/helper.py): persist trained model artifacts.  
- data/ — raw and intermediate CSVs used by the notebook.  
- tests/ — basic smoke tests to validate the pipeline runs in CI.  
- visualisations/ — saved PNG/SVG outputs for dendrograms, PCA projections, metric plots.

---

## Analysis summary (what was done)
1. Data loading and deterministic train/test split via [`helper.load_data`](student-readiness-for-class-course/helper.py).  
2. Preprocessing with deterministic category encodings and numeric conversions:
   - Convert ranges to midpoints with [`helper.to_midpoint`](student-readiness-for-class-course/helper.py).
   - Map "Yes"/"No" to binary.
   - Fill missing values with column means.
   - Scale features using StandardScaler fit on training only (scaler persisted).
   See [`helper.preprocess`](student-readiness-for-class-course/helper.py).
3. Agglomerative clustering (Ward linkage) applied to training features. Model persisted with [`helper.save_trained_model`](student-readiness-for-class-course/helper.py).
4. Hyperparameter sweep across n_clusters (e.g., 2–7) with evaluation metrics:
   - Silhouette Score (higher better)
   - Davies–Bouldin Index (lower better)
   - Calinski–Harabasz Index (higher better)
5. Consensus ranking:
   - Rank each metric per candidate, compute average rank.
   - Normalize metrics to 0–1 where higher is better and compute combined score for tie-breaking.
   - Select recommended n_clusters by Avg_Rank (then combined_score).
6. Prediction for held-out test rows:
   - Transform test rows using saved scaler & category maps.
   - Compute training centroids and assign test rows to nearest centroid (Euclidean).
   - Visualize training vs test points in PCA-projected shared space (PCA fit on training only).
7. Outputs:
   - Clustered examples and counts exported (CSV), plots saved under visualisations.

---

## How to reproduce (local / dev container)
1. Open dev container (Debian 12).  
2. Install dependencies:
   ```sh
   pip install -r requirements.txt  # if present; otherwise: pandas scikit-learn matplotlib seaborn joblib