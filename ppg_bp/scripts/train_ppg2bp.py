import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import sys

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import M5_fusion_transformer

def main():
    # Load preprocessed data
    data_path = "ppg_bp/ppgbp_train.npz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run build_ppgbp_dataset.py first.")
        return
    
    print("Loading data...")
    data = np.load(data_path)
    
    # Prepare tensors
    ppg = torch.tensor(data["x"]).unsqueeze(1)  # (N,1,512)
    
    # Age and BMI need to be broadcast to match temporal dimension after pooling
    # After conv+pool operations, 512 -> 256 -> 128, so we need (N,1,128)
    age_vals = torch.tensor(data["age"]).unsqueeze(1).unsqueeze(2)  # (N,1,1)
    bmi_vals = torch.tensor(data["bmi"]).unsqueeze(1).unsqueeze(2)  # (N,1,1)
    
    # Repeat to match temporal dimension (128)
    age = age_vals.repeat(1, 1, 128)  # (N,1,128)  
    bmi = bmi_vals.repeat(1, 1, 128)  # (N,1,128)
    
    # Normalize BP targets to [0,1] range for training
    sbp = data["sbp"]
    dbp = data["dbp"]
    sbp_min, sbp_max = 80, 200  # reasonable BP range
    dbp_min, dbp_max = 40, 120
    
    sbp_norm = (sbp - sbp_min) / (sbp_max - sbp_min)
    dbp_norm = (dbp - dbp_min) / (dbp_max - dbp_min)
    
    y = torch.tensor(np.stack([sbp_norm, dbp_norm], axis=1))  # (N,2)
    
    print(f"Dataset: {len(ppg)} segments")
    print(f"PPG shape: {ppg.shape}")
    print(f"Age shape: {age.shape}")
    print(f"BMI shape: {bmi.shape}")
    print(f"SBP range: {sbp.min():.1f} - {sbp.max():.1f} mmHg")
    print(f"DBP range: {dbp.min():.1f} - {dbp.max():.1f} mmHg")
    
    # Create dataset and split
    dataset = TensorDataset(ppg, age, bmi, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Setup model and training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = M5_fusion_transformer(n_input=1, n_output=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.L1Loss()  # MAE loss
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(30):
        # Training
        model.train()
        train_losses = []
        
        for ppg_batch, age_batch, bmi_batch, y_batch in train_loader:
            ppg_batch = ppg_batch.to(device)
            age_batch = age_batch.to(device)
            bmi_batch = bmi_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(ppg_batch, age_batch, bmi_batch).squeeze()
            loss = criterion(output, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for ppg_batch, age_batch, bmi_batch, y_batch in val_loader:
                ppg_batch = ppg_batch.to(device)
                age_batch = age_batch.to(device)
                bmi_batch = bmi_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(ppg_batch, age_batch, bmi_batch).squeeze()
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1:2d}/30  Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("ppg_bp/output", exist_ok=True)
            torch.save(model.state_dict(), "ppg_bp/output/ppg2bp_custom.pth")
            print(f"  → Saved new best model (val_loss: {val_loss:.4f})")
    
    print(f"\n✓ Training completed!")
    print(f"✓ Best model saved to ppg_bp/output/ppg2bp_custom.pth")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    
    # Save normalization constants for inference
    norm_params = {
        'sbp_min': float(sbp_min), 'sbp_max': float(sbp_max),
        'dbp_min': float(dbp_min), 'dbp_max': float(dbp_max)
    }
    np.save("ppg_bp/output/bp_norm_params.npy", norm_params)
    print("✓ Saved normalization parameters to ppg_bp/output/bp_norm_params.npy")

if __name__ == "__main__":
    main() 