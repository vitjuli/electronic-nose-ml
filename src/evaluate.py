import argparse, os, json, glob, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from .dataset import split_dataframe, build_matrices, load_scaler
from .model import MLP
from sklearn.metrics import r2_score as sk_r2

def load_ckpts(arg):
    if os.path.isdir(arg):
        man=os.path.join(arg,'ensemble.json')
        if os.path.exists(man): return json.load(open(man))['checkpoints']
        return sorted(glob.glob(os.path.join(arg,'best_model_*.pt')))
    return [arg]

def scatter_plot(y_true, y_pred, title, out_path):
    import matplotlib.pyplot as plt
    plt.figure(); plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo,hi=min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('True'); plt.ylabel('Predicted'); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    df=pd.read_csv(args.data); feat=[c for c in df.columns if c.startswith('x')]; tg=args.targets
    _,_,te=split_dataframe(df, split_col='split'); Xte,Yte=build_matrices(te, feat, tg)
    sc=load_scaler(args.scaler); Xte=sc.transform(Xte).astype(np.float32)
    preds=[]
    with torch.no_grad():
        for pth in load_ckpts(args.ckpt):
            ckpt=torch.load(pth, map_location=device); m=MLP(ckpt['in_dim'], ckpt['hidden'], ckpt['out_dim']); m.load_state_dict(ckpt['state_dict']); m.to(device).eval()
            preds.append(m(torch.from_numpy(Xte).to(device)).cpu().numpy())
    P=np.mean(preds,0); mse=((Yte-P)**2).mean(0); r2=np.array([sk_r2(Yte[:,i],P[:,i]) for i in range(Yte.shape[1])])
    print('Per-target MSE:', mse); print('Per-target R2:', r2); print('Avg MSE:', mse.mean(), '| Avg R2:', r2.mean())
    for i, tgt in enumerate(tg):
        out=os.path.join(os.path.dirname(args.ckpt) if os.path.isdir(args.ckpt) else '.', f'pred_vs_true_{tgt}.png')
        scatter_plot(Yte[:,i], P[:,i], f'Pred vs True — {tgt}', out); print('Saved plot:', out)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--ckpt', required=True); p.add_argument('--scaler', required=True); p.add_argument('--targets', nargs='+', default=['H2_ppm','Propane_ppm']); p.add_argument('--cpu', action='store_true'); args=p.parse_args(); main(args)
