
import yaml
import tensorflow as tf
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

from mmfi_lib.mmfi import make_dataset, make_dataloader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from mmfi_lib.evaluate import calulate_error, compute_pck_pckh
from posenet_model import posenet, weights_init

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_keypoints(pred_keypoints, gt_keypoints, save_dir='visualizations'):
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(3, pred_keypoints.shape[0])
    
    skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # nose-eyes, eyes-ears
        # Upper body
        [5, 6],  # shoulders
        [5, 7], [7, 9],  # left arm
        [6, 8], [8, 10],  # right arm
        # Torso
        [5, 11], [6, 12],  # shoulders to hips
        [11, 12],  # hips
        # Lower body
        [11, 13], [13, 15],  # left leg
        [12, 14], [14, 16]   # right leg
    ]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        pred = pred_keypoints[sample_idx]  # (17, 2)
        gt = gt_keypoints[sample_idx]  # (17, 2)
        
        ax1 = axes[sample_idx, 0]
        ax1.scatter(gt[:, 0], -gt[:, 1], c='blue', s=150, alpha=0.7, label='GT Keypoints', zorder=3)
        for connection in skeleton:
            if connection[0] < len(gt) and connection[1] < len(gt):
                ax1.plot([gt[connection[0], 0], gt[connection[1], 0]], 
                        [-gt[connection[0], 1], -gt[connection[1], 1]], 
                        'b-', linewidth=3, alpha=0.6, zorder=2)
        ax1.set_title(f'Sample {sample_idx+1}: Ground Truth Pose', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        ax2 = axes[sample_idx, 1]
        ax2.scatter(pred[:, 0], -pred[:, 1], c='red', s=150, alpha=0.7, label='Predicted Keypoints', zorder=3)
        for connection in skeleton:
            if connection[0] < len(pred) and connection[1] < len(pred):
                ax2.plot([pred[connection[0], 0], pred[connection[1], 0]], 
                        [-pred[connection[0], 1], -pred[connection[1], 1]], 
                        'r-', linewidth=3, alpha=0.6, zorder=2)
        ax2.set_title(f'Sample {sample_idx+1}: Predicted Pose', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        ax3 = axes[sample_idx, 2]
        ax3.scatter(gt[:, 0], -gt[:, 1], c='blue', s=150, alpha=0.6, label='Ground Truth', zorder=3)
        ax3.scatter(pred[:, 0], -pred[:, 1], c='red', s=150, alpha=0.6, label='Prediction', zorder=3)
        for connection in skeleton:
            if connection[0] < len(gt) and connection[1] < len(gt):
                ax3.plot([gt[connection[0], 0], gt[connection[1], 0]], 
                        [-gt[connection[0], 1], -gt[connection[1], 1]], 
                        'b-', linewidth=2, alpha=0.4, zorder=1)
                ax3.plot([pred[connection[0], 0], pred[connection[1], 0]], 
                        [-pred[connection[0], 1], -pred[connection[1], 1]], 
                        'r-', linewidth=2, alpha=0.4, zorder=2)
        ax3.set_title(f'Sample {sample_idx+1}: Overlay Comparison', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X coordinate')
        ax3.set_ylabel('Y coordinate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/best_model_poses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Best model pose visualization saved to: {save_dir}/best_model_poses.png")
    
set_seed(42)

#X = torch.rand(32,3,114,10)
metafi = posenet()
#metafi = metafi.cuda()
#flops, params = thop.profile(metafi, inputs=(X,))
#print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
#print(f"Parameters: {params / 1e6} Million")

dataset_root =r'D:\NCKH.2025-2026\VinWifi\MMFiDataset'
with open('config.yaml', 'r') as fd:  # change the .yaml file in your code.
    config = yaml.load(fd, Loader=yaml.FullLoader)

train_dataset, test_dataset = make_dataset(dataset_root, config)
rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])

val_data , test_data = train_test_split(test_dataset, test_size=0.5, random_state=41)
val_loader = make_dataloader(val_data, is_training=False, generator=rng_generator, **config['val_loader'])
test_loader = make_dataloader(test_data, is_training=False, generator=rng_generator, **config['test_loader'])

metafi.apply(weights_init)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    metafi = metafi.cuda()
    criterion_L2 = nn.MSELoss().cuda()
    print("Model and loss moved to GPU")
    print(f"Model is on CUDA: {next(metafi.parameters()).is_cuda}")
else:
    device = torch.device("cpu")
    print("WARNING: CUDA not available, using CPU")
    metafi = metafi.to(device)
    criterion_L2 = nn.MSELoss()

#l2_loss = nn.L2Loss().cuda() 
optimizer = torch.optim.SGD(metafi.parameters(), lr = 0.001, momentum=0.9)
n_epochs = 20
n_epochs_decay = 30
epoch_count = 1

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1))

l2_lambda = 0.001
regularization_loss = 0
for param in metafi.parameters():
    regularization_loss += torch.norm(param, p=2)  # L2 regularization term

# Tổng loss function là tổng của hàm loss và regularization loss
def total_loss(output, target):
    loss = criterion_L2(output, target)
    reg_loss = l2_lambda * regularization_loss
    return loss + reg_loss

num_epochs = 50
pck_50_overall_max = 0
train_mean_loss_iter = []
valid_mean_loss_iter = []
time_iter = []

pck_50_epoch_history = []
pck_20_epoch_history = []
mpjpe_epoch_history = []
pa_mpjpe_epoch_history = []

best_pred_keypoints = None
best_gt_keypoints = None

vis_dir = 'visualizations'
os.makedirs(vis_dir, exist_ok=True)
print(f"\nVisualizations will be saved to: {vis_dir}/")

for epoch_index in range(num_epochs):

    loss = 0
    train_loss_iter = []
    metric = []
    metafi.train()
    relation_mean =[]
    if epoch_index == 0 and torch.cuda.is_available():
        print(f"\nEpoch {epoch_index} - GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    for idx, data in enumerate(train_loader):
        csi_data = data['input_wifi-csi']
        
        if not isinstance(csi_data, torch.Tensor):
            csi_data = torch.from_numpy(csi_data).float()
        else:
            csi_data = csi_data.float()
        csi_data = csi_data.to(device)
        
        #csi_dafeaturesta = csi_data.view(16,2,3,114,10)
        keypoint = data['output']#17,3
        if not isinstance(keypoint, torch.Tensor):
            keypoint = torch.from_numpy(keypoint).float()
        keypoint = keypoint.to(device)
       
        xy_keypoint = keypoint[:,:,0:2].to(device)
        confidence = keypoint[:,:,2:3].to(device)

        pred_xy_keypoint, time = metafi(csi_data) #b,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()
        
        # Debug: Verify data is on GPU (only print once)
        if idx == 0 and epoch_index == 0 and torch.cuda.is_available():
            print(f"\nFirst batch - CSI data on CUDA: {csi_data.is_cuda}")
            print(f"First batch - Prediction on CUDA: {pred_xy_keypoint.is_cuda}")
            print(f"GPU Memory allocated after forward: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB\n")
        
        #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 1, 2)
        #flops, params = thop.profile(metafi, inputs=(csi_data,))
        #loss = tf.reduce_mean(tf.pow(pred_xy_keypoint - xy_keypoint, 2))
        #loss = criterion_L2(pred_xy_keypoint, xy_keypoint)/32
        loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/32
        #loss += l2_loss
        train_loss_iter.append(loss.cpu().detach().numpy())
        time_iter.append(time)
        optimizer.zero_grad()

        loss.backward() # retain_graph=True
        optimizer.step()

        lr = np.array(scheduler.get_last_lr())
        #print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
        #print(f"Parameters: {params / 1e6} Million")
        message = '(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) ' % (epoch_index, idx * 32, lr, loss)
        print(message)
    scheduler.step()
    sum_time = np.mean(time_iter)
    train_mean_loss = np.mean(train_loss_iter)
    train_mean_loss_iter.append(train_mean_loss)
    # relation_mean = np.mean(relation, 0)
    print('end of the epoch: %d, with loss: %.3f' % (epoch_index, train_mean_loss,))
    #total_params = sum(p.numel() for p in metafi.parameters())
    #print("Số lượng tham số trong mô hình: ", total_params)
    #print("Tổng thời gian train: ", sum_time)
    
    metafi.eval()
    valid_loss_iter = []
    #metric = []
    pck_50_iter = []
    pck_20_iter = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            csi_data = data['input_wifi-csi']
            keypoint = data['output']
            
            if not isinstance(csi_data, torch.Tensor):
                csi_data = torch.from_numpy(csi_data)
            csi_data = csi_data.float().to(device, non_blocking=True)  # ADD non_blocking

            if not isinstance(keypoint, torch.Tensor):
                keypoint = torch.from_numpy(keypoint)
            keypoint = keypoint.float().to(device, non_blocking=True)  # ADD non_blocking

            xy_keypoint = keypoint[:, :, 0:2]      # Already on GPU
            confidence = keypoint[:, :, 2:3] 

            pred_xy_keypoint,time = metafi(csi_data)  # 4,2,17,17
            #pred_xy_keypoint = pred_xy_keypoint.squeeze()
            #pred_xy_keypoint = pred_xy_keypoint.reshape(length_val,17,2)
            #flops, params = thop.profile(metafi, inputs=(csi_data,))
            
            #loss = tf.reduce_mean(tf.pow(pred_xy_keypoint - xy_keypoint, 2))
            loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))
            #loss = criterion_L2(pred_xy_keypoint, xy_keypoint)
            
            valid_loss_iter.append(loss.cpu().detach().numpy())
            pred_xy_keypoint = pred_xy_keypoint.cpu().detach().numpy()
            xy_keypoint = xy_keypoint.cpu().detach().numpy()
            #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1).unsqueeze(dim=0)
            #xy_keypoint = torch.transpose(xy_keypoint, 0, 1).unsqueeze(dim=0)
            pred_xy_keypoint_pck = np.transpose(pred_xy_keypoint, (0, 2, 1))
            xy_keypoint_pck = np.transpose(xy_keypoint, (0, 2, 1))
            #keypoint = torch.transpose(keypoint, 1, 2)
            #pred_xy_keypoint_pck = pred_xy_keypoint.cpu()
            #xy_keypoint_pck = xy_keypoint.cpu()
            pck = compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5)
            #mpjpe,pa_mpjpe = calulate_error(pred_xy_keypoint, xy_keypoint)
             
            metric.append(calulate_error(pred_xy_keypoint, xy_keypoint))
            pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.5))
            pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint_pck, xy_keypoint_pck, 0.2))
            
            # Store first batch predictions for best model visualization
            if idx == 0:
                current_batch_pred = pred_xy_keypoint.copy()
                current_batch_gt = xy_keypoint.copy()
            
            #message1 = '( loss: %.3f) ' % (loss)
            #print(message1)
            #pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5))
            #print(f"FLOPs: {flops / 1e9} GigaFLOPs")  # Chuyển đổi sang GigaFLOPs (tỷ lệ FLOPs)
            #print(f"Parameters: {params / 1e6} Million")
            #pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.2))

        valid_mean_loss = np.mean(valid_loss_iter)
        #train_mean_loss = np.mean(train_loss_iter)
        valid_mean_loss_iter.append(valid_mean_loss)
        mean = np.mean(metric, 0)*1000
        mpjpe_mean = mean[0]
        pa_mpjpe_mean = mean[1]
        pck_50 = np.mean(pck_50_iter,0)
        pck_20 = np.mean(pck_20_iter,0)
        pck_50_overall = pck_50[17]
        pck_20_overall = pck_20[17]
        
        # Store metrics for plotting
        pck_50_epoch_history.append(pck_50_overall)
        pck_20_epoch_history.append(pck_20_overall)
        mpjpe_epoch_history.append(mpjpe_mean)
        pa_mpjpe_epoch_history.append(pa_mpjpe_mean)
        
        print('validation result with loss: %.3f, pck_50: %.3f, pck_20: %.3f, mpjpe: %.3f, pa_mpjpe: %.3f' % (valid_mean_loss, pck_50_overall,pck_20_overall, mpjpe_mean, pa_mpjpe_mean))
        
        if pck_50_overall > pck_50_overall_max:
           print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
           torch.save(metafi, 'bestmodel.pth')
           pck_50_overall_max = pck_50_overall
           # Store best model predictions for final visualization
           best_pred_keypoints = current_batch_pred.copy()
           best_gt_keypoints = current_batch_gt.copy()

        if (epoch_index+1) % 50 == 0:
            print('the train loss for the first %.1f epoch is' % (epoch_index))
            print(train_mean_loss_iter)

print("\n" + "="*60)
print("TRAINING COMPLETED - Generating Final Visualizations")
print("="*60)

if best_pred_keypoints is not None and best_gt_keypoints is not None:
    print("\nGenerating pose visualization for best model...")
    visualize_keypoints(best_pred_keypoints, best_gt_keypoints, save_dir=vis_dir)
else:
    print("\nWarning: No best model predictions saved for visualization")

# Print final statistics
print("\n" + "="*60)
print("FINAL TRAINING STATISTICS")
print("="*60)
print(f"Best PCK@0.5: {pck_50_overall_max:.3f}%")
print(f"Final Training Loss: {train_mean_loss_iter[-1]:.6f}")
print(f"Final Validation Loss: {valid_mean_loss_iter[-1]:.6f}")
print(f"Final PCK@0.5: {pck_50_epoch_history[-1]:.3f}%")
print(f"Final PCK@0.2: {pck_20_epoch_history[-1]:.3f}%")
print(f"Final MPJPE: {mpjpe_epoch_history[-1]:.3f} mm")
print(f"Final PA-MPJPE: {pa_mpjpe_epoch_history[-1]:.3f} mm")
print("="*60)

# Create a simple loss comparison plot
plt.figure(figsize=(10, 6))

epochs = list(range(1,num_epochs+1))
training_loss = train_mean_loss_iter
validation_loss = valid_mean_loss_iter

plt.plot(epochs, training_loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(epochs, validation_loss, label='Validation Loss', color='red', linewidth=2)

plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Loss Function Over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{vis_dir}/simple_loss_plot.png', dpi=150, bbox_inches='tight')
print(f"\nSimple loss plot saved to: {vis_dir}/simple_loss_plot.png")
plt.show()

print(f"\n{'='*60}")
print(f"All visualizations saved in folder: {vis_dir}/")
print(f"{'='*60}\n")