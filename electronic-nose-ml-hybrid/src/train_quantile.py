import argparse, os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .dataset import split_dataframe, build_matrices, fit_scale_save
from .losses import pinball_loss
from .model import MLP

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data); feature_cols=[c for c in df.columns if c.startswith('x')]; target_cols=args.targets
    tr, va, _ = split_dataframe(df, split_col='split')
    Xtr, Ytr = build_matrices(tr, feature_cols, target_cols); Xva, Yva = build_matrices(va, feature_cols, target_cols)
    sc = fit_scale_save(Xtr, os.path.join(args.outdir, 'scaler_quantile.joblib'))
    Xtr = sc.transform(Xtr).astype(np.float32); Xva = sc.transform(Xva).astype(np.float32)
    Q = [float(q) for q in args.quantiles]; out_dim = len(target_cols)*len(Q)
    model = MLP(Xtr.shape[1], args.hidden, out_dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), batch_size=args.batch_size, shuffle=False)
    best=float('inf'); pc=0
    for ep in range(1, args.epochs+1):
        model.train(); tot=0
        for xb,yb in tr_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = pinball_loss(model(xb), yb, Q); loss.backward(); opt.step(); tot += loss.item()*xb.size(0)
        # val
        model.eval(); v=0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb,yb = xb.to(device), yb.to(device); v += pinball_loss(model(xb), yb, Q).item()*xb.size(0)
        v/=len(va_loader.dataset)
        print(f'Epoch {ep:03d} | val_pinball={v:.4f}')
        if v<best-1e-8: best=v; pc=0; torch.save({'state_dict': model.state_dict(),'in_dim':Xtr.shape[1],'hidden':args.hidden,'out_dim':out_dim,'quantiles':Q,'target_cols':target_cols}, os.path.join(args.outdir,'best_quantile.pt'))
        else:
            pc+=1; if pc>=40: print('Early stopping.'); break
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--outdir', default='models'); p.add_argument('--targets', nargs='+', default=['H2_ppm','Propane_ppm']); p.add_argument('--quantiles', nargs='+', default=['0.05','0.5','0.95']); p.add_argument('--epochs', type=int, default=200); p.add_argument('--batch_size', type=int, default=128); p.add_argument('--hidden', type=int, default=64); p.add_argument('--dropout', type=float, default=0.1); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--cpu', action='store_true'); args=p.parse_args(); main(args)
