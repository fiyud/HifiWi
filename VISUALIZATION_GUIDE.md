# Training Visualization Guide

This guide explains the visualizations generated during training of the pose estimation model.

## Visualization Types

### 1. Keypoint Predictions (`keypoints_epoch{X}_batch{Y}.png`)
**Generated:** First batch of epochs 0-4, then every 10th epoch  
**Location:** `visualizations/`

Shows three side-by-side comparisons:
- **Left Panel:** Ground truth keypoints with skeleton connections (blue)
- **Middle Panel:** Predicted keypoints with skeleton connections (red)  
- **Right Panel:** Overlay of both for easy comparison

**What to look for:**
- Predictions should gradually align better with ground truth
- Skeleton structure should remain consistent
- Watch for outlier keypoints or broken skeleton connections

---

### 2. Training Progress Dashboard (`training_progress_{timestamp}.png`)
**Generated:** Every 5 epochs and at the end of training  
**Location:** `visualizations/`

A comprehensive 6-panel dashboard showing:

#### Panel 1 (Top, Full Width): Training & Validation Loss
- Blue line: Training loss over epochs
- Red line: Validation loss over epochs
- **Watch for:** Overfitting (validation loss increasing while training decreases)

#### Panel 2 (Middle Left): PCK@0.5
- Green line: Percentage of Correct Keypoints at 50% threshold
- Range: 0-100%
- **Higher is better** - target: >90%

#### Panel 3 (Middle Right): PCK@0.2
- Magenta line: Percentage of Correct Keypoints at 20% threshold (stricter)
- Range: 0-100%
- **Higher is better** - target: >70%

#### Panel 4 (Bottom Left): MPJPE (Mean Per Joint Position Error)
- Cyan line: Average euclidean distance error in pixels
- **Lower is better** - measures absolute position accuracy

#### Panel 5 (Bottom Right): PA-MPJPE (Procrustes Aligned MPJPE)
- Yellow line: Error after optimal alignment (rotation/scale/translation)
- **Lower is better** - measures pose structure accuracy independent of position

---

### 3. Simple Loss Plot (`simple_loss_plot.png`)
**Generated:** At end of training  
**Location:** `visualizations/`

Clean comparison of training vs validation loss
- Use for presentations or quick reference
- Same data as Panel 1 of the dashboard

---

## Console Output

### During Training (Each Epoch)
```
(epoch: X, iters: Y, lr: Z, loss: W)
```

### After Each Epoch
```
end of the epoch: X, with loss: Y
validation result with loss: A, pck_50: B, pck_20: C, mpjpe: D, pa_mpjpe: E
```

### GPU Information (First Epoch)
```
Using GPU: [GPU Name]
GPU Memory: [XX.XX] GB
Model is on CUDA: True
Epoch 0 - GPU Memory allocated: [XX.XX] MB
First batch - CSI data on CUDA: True
First batch - Prediction on CUDA: True
```

### Final Statistics
```
============================================================
FINAL TRAINING STATISTICS
============================================================
Best PCK@0.5: XX.XXX%
Final Training Loss: X.XXXXXX
Final Validation Loss: X.XXXXXX
Final PCK@0.5: XX.XXX%
Final PCK@0.2: XX.XXX%
Final MPJPE: XXX.XXX mm
Final PA-MPJPE: XXX.XXX mm
============================================================
```

---

## Metrics Explained

### PCK (Percentage of Correct Keypoints)
- Measures how many predicted keypoints are within a certain threshold of ground truth
- Threshold is normalized by torso diameter (distance between shoulder and hip)
- PCK@0.5 = 50% of torso diameter, PCK@0.2 = 20% of torso diameter

### MPJPE (Mean Per Joint Position Error)
- Average euclidean distance between predicted and ground truth keypoints
- Measured in millimeters (mm) after scaling
- Sensitive to global position errors

### PA-MPJPE (Procrustes Aligned MPJPE)
- MPJPE after optimal rigid alignment (Procrustes analysis)
- Removes global translation, rotation, and scaling
- Better measure of pose structure quality
- Usually lower than MPJPE

---

## Tips for Monitoring Training

1. **Check GPU usage first:** Verify GPU is being used with the console output
2. **Watch validation loss:** Should decrease and stay close to training loss
3. **PCK metrics:** Should steadily increase, especially PCK@0.5
4. **Keypoint visualizations:** Predictions should visibly improve over epochs
5. **Early stopping:** If validation metrics don't improve for many epochs, consider stopping

---

## File Organization

```
ECCV24_Li-HPE/
├── visualizations/
│   ├── keypoints_epoch0_batch0.png
│   ├── keypoints_epoch1_batch0.png
│   ├── ...
│   ├── training_progress_20251025_HHMMSS.png
│   ├── training_progress_20251025_HHMMSS.png (updates every 5 epochs)
│   └── simple_loss_plot.png
├── trainer.py
└── VISUALIZATION_GUIDE.md (this file)
```

---

## Troubleshooting

### No visualizations appearing?
- Check if `visualizations/` folder is created
- Ensure matplotlib and seaborn are installed: `pip install matplotlib seaborn`

### Plots look wrong?
- Check data shapes in console output
- Verify keypoint indices match your dataset (currently using COCO 17-keypoint format)

### GPU not being used?
- Look for "WARNING: CUDA not available" message
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
