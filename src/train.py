import argparse, os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .dataset import split_dataframe, build_matrices, fit_scale_save
from .model import MLP
from .utils import set_all_seeds

def r2_avg(Y, P):
    ss_res = ((Y-P)**2).sum(0); ss_tot = ((Y-Y.mean(0))**2).sum(0); return float(np.mean(1 - ss_res/np.maximum(ss_tot,1e-12)))

def train_epoch(model, loader, opt, crit, device):
    model.train(); tot=0.0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad(); pr=model(xb); loss=crit(pr,yb); loss.backward(); opt.step(); tot+=loss.item()*xb.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval(); tot=0.0; Y=[]; P=[]
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device); pr=model(xb); tot+=crit(pr,yb).item()*xb.size(0); Y.append(yb.cpu().numpy()); P.append(pr.cpu().numpy())
    Y=np.concatenate(Y,0); P=np.concatenate(P,0); return tot/len(loader.dataset), float(((Y-P)**2).mean()), r2_avg(Y,P)

def train_single(args, seed_offset=0, idx=0):
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_all_seeds(args.seed+seed_offset); os.makedirs(args.outdir, exist_ok=True)
    df=pd.read_csv(args.data); feat=[c for c in df.columns if c.startswith('x')]; tg=args.targets
    tr,va,te = split_dataframe(df, split_col='split', train_size=args.train_size, val_size=args.val_size, test_size=1.0-args.train_size-args.val_size, random_state=args.seed)
    Xtr,Ytr=build_matrices(tr, feat, tg); Xva,Yva=build_matrices(va, feat, tg); Xte,Yte=build_matrices(te, feat, tg)
    sc=fit_scale_save(Xtr, os.path.join(args.outdir,'scaler.joblib'))
    Xtr=sc.transform(Xtr).astype(np.float32); Xva=sc.transform(Xva).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    tr_loader=DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=args.batch_size, shuffle=True)
    va_loader=DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), batch_size=args.batch_size, shuffle=False)
    te_loader=DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(Yte)), batch_size=args.batch_size, shuffle=False)
    model=MLP(Xtr.shape[1], args.hidden, len(tg), args.dropout).to(device); opt=torch.optim.Adam(model.parameters(), lr=args.lr); crit=nn.MSELoss()
    best=1e18; pc=0; path=os.path.join(args.outdir, f'best_model_{idx}.pt')
    for ep in range(1, args.epochs+1):
        trl=train_epoch(model,tr_loader,opt,crit,device); val, mse, r2=eval_epoch(model,va_loader,crit,device)
        print(f'Epoch {ep:03d} | train={trl:.4f} | val={val:.4f} | mse={mse:.4f} | r2={r2:.4f}')
        if val<best-1e-8: best=val; pc=0; torch.save({'state_dict':model.state_dict(),'in_dim':Xtr.shape[1],'hidden':args.hidden,'out_dim':len(tg),'feature_cols':feat,'target_cols':tg}, path)
        else:
            pc+=1; if pc>=args.patience: print('Early stopping.'); break
    # final test
    ckpt=torch.load(path, map_location='cpu'); model.load_state_dict(ckpt['state_dict'])
    te_loss, te_mse, te_r2 = eval_epoch(model, te_loader, crit, device)
    print(f'Test | loss={te_loss:.4f} | mse={te_mse:.4f} | r2={te_r2:.4f}')
    return path

def main(args):
    os.makedirs(args.outdir, exist_ok=True); ckpts=[]
    for i in range(args.n_models):
        print(f'=== Model {i+1}/{args.n_models} (seed={args.seed+i}) ==='); ckpts.append(train_single(args, seed_offset=i, idx=i))
    with open(os.path.join(args.outdir,'ensemble.json'),'w',encoding='utf-8') as f:
        json.dump({'checkpoints':ckpts,'scaler':os.path.join(args.outdir,'scaler.joblib'),'targets':args.targets}, f, indent=2)
    print('Ensemble manifest saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--outdir', default='models'); p.add_argument('--targets', nargs='+', default=['H2_ppm','Propane_ppm']); p.add_argument('--epochs', type=int, default=400); p.add_argument('--patience', type=int, default=60); p.add_argument('--batch_size', type=int, default=128); p.add_argument('--hidden', type=int, default=32); p.add_argument('--dropout', type=float, default=0.0); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--train_size', type=float, default=0.6); p.add_argument('--val_size', type=float, default=0.2); p.add_argument('--seed', type=int, default=42); p.add_argument('--n_models', type=int, default=1); p.add_argument('--cpu', action='store_true'); args=p.parse_args(); main(args)
