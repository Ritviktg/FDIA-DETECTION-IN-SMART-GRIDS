import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import rtdl
import joblib
import os
import warnings
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

warnings.filterwarnings('ignore', category=UserWarning)

def train_ft_transformer(grid_name, n_epochs=150,
                         batch_size=256, lr=1e-4,
                         random_state=42):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    X_train = np.load(f'dataset/X_train_{grid_name}.npy')
    X_test  = np.load(f'dataset/X_test_{grid_name}.npy')
    y_train = np.load(f'dataset/y_train_{grid_name}.npy')
    y_test  = np.load(f'dataset/y_test_{grid_name}.npy')
    
    n_features = X_train.shape[1]
    
    # Keeping class weights as 1.0 for both classes because the data is perfectly balanced
    class_weight = torch.tensor(
        [1.0, 1.0], dtype=torch.float32
    ).to(device)
    
    # Configure baseline hyperparameters based on grid
    if grid_name == '14':
        d_token = 64
        ffn_d_hidden = 128
    else:  # 118-bus
        d_token = 128
        ffn_d_hidden = 256
        
    print(f"\nGrid {grid_name}-bus FT-Transformer training:")
    print(f"  Features:        {n_features}")
    print(f"  Train samples:   {len(X_train)}")
    print(f"  Batch size:      {batch_size}")
    print(f"  d_token:         {d_token}")
    print(f"  ffn_d_hidden:    {ffn_d_hidden}")
    print(f"  Device:          {device}")
    
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_token=d_token,
        n_blocks=2,
        attention_dropout=0.2,
        ffn_d_hidden=ffn_d_hidden,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        last_layer_query_idx=[-1],
        d_out=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )
    
    # Plateau learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=10, verbose=True
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, 
                         batch_size=batch_size, 
                         shuffle=True)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 0.001
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out  = model(X_batch, None)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{n_epochs} — loss: {avg_epoch_loss:.4f}")
        
        # Step the scheduler using training loss
        scheduler.step(avg_epoch_loss)
        
        # Early Stopping check
        if avg_epoch_loss < best_loss - min_delta:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= 20:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
    
    model.eval()
    y_pred_list = []
    eval_dataset = TensorDataset(X_test_t)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (X_batch,) in eval_loader:
            logits = model(X_batch, None)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_list.append(preds)
    y_pred = np.concatenate(y_pred_list)
    
    acc = round(accuracy_score(y_test, y_pred),  4)
    pre = round(precision_score(y_test, y_pred,
                                zero_division=0), 4)
    rec = round(recall_score(y_test, y_pred,
                             zero_division=0), 4)
    f1  = round(f1_score(y_test, y_pred,
                         zero_division=0), 4)
    
    print(f"\nFT-Transformer results on {grid_name}-bus:")
    print(f"  ACC: {acc}")
    print(f"  PRE: {pre}")
    print(f"  REC: {rec}")
    print(f"  F1:  {f1}")
    
    os.makedirs('models', exist_ok=True)
    
    # Save the model state on CPU for portability and downstream loading compatibility
    model.cpu()
    torch.save(model.state_dict(),
               f'models/ft_transformer_{grid_name}.pt')
    joblib.dump(model,
                f'models/ft_transformer_{grid_name}.pkl')
    
    np.save(f'dataset/y_pred_ft_{grid_name}.npy', y_pred)
    
    return model, y_pred, {
        'ACC': acc, 'PRE': pre, 'REC': rec, 'F1': f1
    }

if __name__ == '__main__':
    model_14,  y_pred_14,  results_ft_14  = train_ft_transformer('14', n_epochs=150, batch_size=256)
    model_118, y_pred_118, results_ft_118 = train_ft_transformer('118', n_epochs=150, batch_size=64)
    print("\nFT-Transformer training complete for both grids.")
