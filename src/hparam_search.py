import argparse, os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .dataset import split_dataframe, build_matrices, fit_scale_save
from .model import MLP

def main(args):
    import optuna
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    df = pd.read_csv(args.data); feature_cols=[c for c in df.columns if c.startswith('x')]; target_cols=args.targets
    def objective(trial):
        hidden = trial.suggest_int('hidden', 16, 128, step=16)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        batch = trial.suggest_categorical('batch_size', [64,128,256])
        tr, va, _ = split_dataframe(df, split_col='split')
        Xtr, Ytr = build_matrices(tr, feature_cols, target_cols)
        Xva, Yva = build_matrices(va, feature_cols, target_cols)
        sc = fit_scale_save(Xtr, 'models/search_scaler.joblib')
        Xtr = sc.transform(Xtr).astype(np.float32); Xva = sc.transform(Xva).astype(np.float32)
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=batch, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), batch_size=batch, shuffle=False)
        model = MLP(Xtr.shape[1], hidden, len(target_cols), dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.MSELoss()
        best=float('inf'); pc=0
        for ep in range(200):
            model.train()
            for xb,yb in tr_loader:
                xb,yb = xb.to(device), yb.to(device); opt.zero_grad(); loss=crit(model(xb), yb); loss.backward(); opt.step()
            # val
            model.eval(); mse=0.0; n=0
            with torch.no_grad():
                for xb,yb in va_loader:
                    xb=xb.to(device); pr=model(xb).cpu().numpy(); mse+=float(((pr-yb.numpy())**2).mean())*xb.size(0); n+=xb.size(0)
            mse/=max(n,1)
            if mse<best-1e-8: best=mse; pc=0
            else: pc+=1; if pc>=40: break
        return best
    study = optuna.create_study(direction='minimize'); study.optimize(objective, n_trials=args.trials)
    print('Best params:', study.best_params)
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'optuna_best.json'), 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--outdir', default='models'); p.add_argument('--targets', nargs='+', default=['H2_ppm','Propane_ppm']); p.add_argument('--trials', type=int, default=10); p.add_argument('--cpu', action='store_true'); args = p.parse_args(); main(args)
