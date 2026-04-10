#!/bin/bash
# BraTS 2023 Dataset Download Workflow
# Dataset requires registration at: https://www.synapse.org/Synapse:syn51156910
# After registration, download via Synapse CLI

set -e
DATA_DIR="$(dirname "$0")/../data/raw/brats2023"

echo "=== BraTS 2023 Download Instructions ==="
echo ""
echo "Step 1: Register at https://www.synapse.org and join the BraTS 2023 challenge"
echo "Step 2: Install synapse client:"
echo "        pip install synapseclient"
echo ""
echo "Step 3: Login and download:"
echo "        synapse login"
echo "        synapse get -r syn51156910 --downloadLocation $DATA_DIR"
echo ""
echo "--- Alternative: Download via Kaggle (if available) ---"
echo "        kaggle competitions download -c rsna-2023-brain-tumor-radiogenomic-classification"
echo ""
echo "--- Or use the provided subset for quick testing ---"
echo "        python scripts/download_sample.py"
echo ""
echo "Expected structure after download:"
echo "  data/raw/brats2023/"
echo "  ├── BraTS-GLI-00000-000/"
echo "  │   ├── BraTS-GLI-00000-000-t1c.nii.gz  (T1 contrast)"
echo "  │   ├── BraTS-GLI-00000-000-t1n.nii.gz  (T1 native)"
echo "  │   ├── BraTS-GLI-00000-000-t2f.nii.gz  (T2 FLAIR)"
echo "  │   ├── BraTS-GLI-00000-000-t2w.nii.gz  (T2 weighted)"
echo "  │   └── BraTS-GLI-00000-000-seg.nii.gz  (segmentation GT)"
echo "  └── ..."
