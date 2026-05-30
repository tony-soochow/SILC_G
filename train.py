import os
import sys
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_recall_curve, confusion_matrix,
                             precision_score, recall_score, matthews_corrcoef)

from model.model import DTIModel, MultiLossFunction
from util.dataset import DTIGraphDataset, collate_fn
from util.seed_set import set_seed

def evaluate(model, loader, device):
    model.eval()
    preds, labels, pred_graphs = [], [], []
    with torch.no_grad():
        for batch in loader:
            drug_g, drug_h, prot_g, prot_h, label = [x.to(device) for x in batch]
            output = model(drug_g, drug_h, prot_g, prot_h, label)
            if isinstance(output, dict):
                
                pred = torch.sigmoid(output['main_pred'])
                pred_graph = torch.sigmoid(output['graph_pred'])
            else:
               
                pred = torch.sigmoid(output)
                
            preds.extend(pred.cpu().numpy())
            pred_graphs.extend(pred_graph.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return roc_auc_score(labels, preds), roc_auc_score(labels, pred_graphs)

def train(model, device, train_loader, val_loader, args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = MultiLossFunction(
        alpha=getattr(args, 'alpha', 1.0),
        beta=getattr(args, 'beta', 0.3), 
        gamma=getattr(args, 'gamma', 0.3),
        delta=getattr(args, 'delta', 0.2)
        # delta = args.delta
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_auc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        # total_loss = 0
        total_losses = {'total': 0, 'main': 0, 'graph': 0, 'spatial': 0, 'contrastive': 0}

        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        for batch in train_loader:
            drug_g, drug_h, prot_g, prot_h, label = [x.to(device) for x in batch]
            pred = model(drug_g, drug_h, prot_g, prot_h, label)
            loss_dict = criterion(model, pred, label.float())
            loss = loss_dict['total_loss']
            
            
            total_losses['total'] += loss_dict['total_loss'].item()
            total_losses['main'] += loss_dict['main_loss'].item()
            total_losses['graph'] += loss_dict['graph_loss'].item()
            total_losses['spatial'] += loss_dict['spatial_loss'].item()
            # total_losses['contrastive'] += loss_dict['contrastive_loss'].item()
            total_losses['contrastive'] += loss_dict['contrastive_loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()
                   
        
        avg_losses = {k: v/len(train_loader) for k, v in total_losses.items()}
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  📊 Total Loss: {avg_losses['total']:.4f}")
        print(f"  🎯 Main Loss: {avg_losses['main']:.4f}")
        print(f"  📈 Graph Loss: {avg_losses['graph']:.4f}")
        print(f"  📈 Spatial Loss: {avg_losses['spatial']:.4f}")
        print(f"  📈 contrastive Loss: {avg_losses['contrastive']:.4f}")
         # print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")
        auc, auc_graph = evaluate(model, val_loader, device)
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation auc_graph: {auc_graph:.4f}")
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model!")

    return best_epoch

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
    }

def test(model, test_loader, device, args, save_dir=None, best_epoch=None):
    model.eval()
    preds, labels = [], []
    total_loss = 0
    
    criterion = MultiLossFunction(
        alpha=getattr(args, 'alpha', 1.0),
        beta=getattr(args, 'beta', 0.3),
        gamma=getattr(args, 'gamma', 0.3),
        delta=getattr(args, 'delta', 0)
    )
    
    main_preds, graph_preds, spatial_preds = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            # batch = batch.to(device, non_blocking=True)
            drug_g, drug_h, prot_g, prot_h, label = [x.to(device) for x in batch]
            output = model(drug_g, drug_h, prot_g, prot_h, label)
            loss_dict = criterion(model, output, label.float())
            total_loss += loss_dict['total_loss'].item()
            
            main_prob = torch.sigmoid(output['main_pred'])
            graph_prob = torch.sigmoid(output['graph_pred'])
            spatial_prob = torch.sigmoid(output['spatial_pred'])
            # spatial_prob = torch.sigmoid(output['spatial_pred'])
            # prob = torch.sigmoid(logits)
            preds.extend(main_prob.cpu().numpy())
            labels.extend(label.cpu().numpy())

            main_preds.extend(main_prob.cpu().numpy())
            graph_preds.extend(graph_prob.cpu().numpy())
            spatial_preds.extend(spatial_prob.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    main_preds = np.array(main_preds)
    graph_preds = np.array(graph_preds)
    spatial_preds = np.array(spatial_preds)

    test_loss = total_loss / len(test_loader)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"test_predictions_epoch{best_epoch or 'final'}.csv")

        df_results = pd.DataFrame({
            "label": labels.flatten(),
            "main_pred": main_preds.flatten(),
            "graph_pred": graph_preds.flatten(),
            "spatial_pred": spatial_preds.flatten()
        })

        df_results.to_csv(save_path, index=False)
        print(f"✅ Test predictions saved to: {save_path}")
    
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"🎯 Best Threshold: {best_threshold:.4f}")
    main_metrics = compute_metrics(labels, main_preds, threshold=best_threshold)
    graph_metrics = compute_metrics(labels, graph_preds, threshold=best_threshold)
    spatial_metrics = compute_metrics(labels, spatial_preds, threshold=best_threshold)
    


    
    final_metrics = main_metrics.copy()
    final_metrics.update({
        "Threshold": float(best_threshold),
        "Test_loss": test_loss,
        "Best Epoch": best_epoch if best_epoch is not None else "N/A",
        })

    print("=" * 6)
    print("🎯 MAIN PREDICTION METRICS")
    print("=" * 6)
    for k, v in main_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("\n" + "=" * 6)
    print("📈 AUXILIARY METRICS ANALYSIS")
    print("=" * 6)
    print(f"📊 Graph Prediction AUC: {graph_metrics['AUROC']:.4f}")
   
    print("\n" + "=" * 6)
    print(f"💾 Test Loss: {test_loss:.4f}")
    print(f"🏆 Best Epoch: {best_epoch if best_epoch is not None else 'N/A'}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        
        np.save(os.path.join(save_dir, "main_preds.npy"), main_preds)
        np.save(os.path.join(save_dir, "graph_preds.npy"), graph_preds)
        np.save(os.path.join(save_dir, "test_labels.npy"), labels)
        
        with open(os.path.join(save_dir, "test_result.json"), "w") as f:
            json.dump({
                "timestamp": args.timestamp,
                "best_epoch": best_epoch,
                "params": {k: getattr(args, k) for k in vars(args)},
                "main_metrics": main_metrics,
                "graph_metrics": graph_metrics,
                "spatial_metrics":spatial_metrics,
                "final_metrics": final_metrics
            }, f, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--embedding', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--data_root', type=str, default='data/biosnap/full_data')
    parser.add_argument('--data', type=str, default='biosnap')
    parser.add_argument('--split', type=str, default='full_data')
    parser.add_argument('--test', action='store_true', help='Only run test')
    
    parser.add_argument('--alpha', type=float, default=1.0, help='Main loss weight')
    parser.add_argument('--beta', type=float, default=0.3, help='Graph loss weight')
    parser.add_argument('--gamma', type=float, default=0, help='Spatial loss weight')
    parser.add_argument('--delta', type=float, default=0, help='Similarity loss weight')
    
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for contrastive learning')
    parser.add_argument('--bank_size', type=int, default=500, help='Sample bank size for contrastive learning')
    parser.add_argument('--cluster_update_interval', type=int, default=500, help='Cluster update interval (batches)')
    parser.add_argument("--temperature",type=float,default=0.1,help="temperature coefficient for contrastive learning")
    parser.add_argument("--momentum",type=float,default=0.1,help="momentum coefficient for memory bank update; 0 means direct overwrite")
    args = parser.parse_args()
    
    timestamp = time.strftime("%m%d_%H%M", time.localtime())
    args.timestamp = timestamp
    args.result_dir = f'LLM_various/protein_150M_graphs_pkl/{args.split}/{args.data}/bank_size_{args.bank_size}/test_result_{timestamp}'
    args.save_path = f'{args.result_dir}/best_model.pth'
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_root = f"data/{args.data}/{args.split}"

    if args.test:
        test_set = DTIGraphDataset(data_split='test', root=data_root, split=args.split)
        print(f"Test set size: {len(test_set)}")
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        sample = test_set[0]
        model = DTIModel(
            in_dim_drug=sample[1].shape[1], 
            in_dim_prot=sample[3].shape[1],
            hidden_dim = args.embedding,
            bank_size=args.bank_size,     
            cluster_update_interval = args.cluster_update_interval,
            temperature = args.temperature,
            momentum = args.momentum
        ).to(device)
        model.load_state_dict(torch.load(args.save_path))
        print("Loaded best model for testing.")
        test(model, test_loader, device, args, save_dir=args.result_dir)
    else:
        train_set = DTIGraphDataset(data_split='train', root=data_root, split=args.split)
        val_set = DTIGraphDataset(data_split='val', root=data_root, split=args.split)
        test_set = DTIGraphDataset(data_split='test', root=data_root, split=args.split)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, prefetch_factor=2, pin_memory=True)

        sample = train_set[0]
        
        model = DTIModel(
            in_dim_drug=sample[1].shape[1],
            
            in_dim_prot=sample[3].shape[1],
            hidden_dim = args.embedding,
            
            bank_size=args.bank_size,     
            cluster_update_interval = args.cluster_update_interval,
            temperature = args.temperature,
            momentum = args.momentum
        ).to(device)
        print(next(model.parameters()).device)
        

        best_epoch = train(model, device, train_loader, val_loader, args)
        model.load_state_dict(torch.load(args.save_path))
        print(f"Loaded best model from epoch {best_epoch} for final testing.")
        test(model, test_loader, device, args, save_dir=args.result_dir, best_epoch=best_epoch)