"""
RESEARCH CONTEXT — READ CAREFULLY BEFORE GENERATING CODE

This project implements a privacy-preserving healthcare security framework using
Split Federated Learning on the MIMIC-IV dataset.

Current stage of research:
- The full dataset has been split into multiple clients by patient_id.
- This script is for CLIENT1 only.
- Client1 has a local subset of MIMIC-IV tables stored in the project folder.
- Each client must train a LOCAL ENCODER model on its own data.
- Raw patient data must NEVER leave the client machine.

Goal of this script:
1. Load and preprocess Client1’s local MIMIC-IV tables.
2. Construct patient-level feature vectors suitable for machine learning.
3. Train a LOCAL CLIENT-SIDE ENCODER model.
4. The encoder must output an intermediate activation vector.
5. Later this activation will be sent to a remote server model in split learning.
6. For now, include a temporary classifier head only for local pretraining.

Important constraints:
- Do NOT implement federated averaging.
- Do NOT implement networking.
- Do NOT aggregate data across clients.
- All processing must remain local.
- The model must be modular so the classifier can be removed later.
- Code must be reproducible and suitable for research publication.

Dataset notes:
- The dataset is already split by patient_id for Client1.
- Tables may include patients, admissions, icustays, chartevents, labevents, etc.
- Code should automatically detect available tables in the local folder.
- Preprocessing must handle missing values, normalization, and feature aggregation.
- Output should be one feature vector per patient stay.

Write modular, well-commented, research-quality code.
"""