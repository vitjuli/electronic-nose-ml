import csv, os, json, time

class CSVLogger:
    def __init__(self, outdir: str, filename: str = 'metrics.csv'):
        self.path = os.path.join(outdir, filename); self._init=False; os.makedirs(outdir, exist_ok=True)
    def log(self, row):
        row = dict(row); row.setdefault('timestamp', int(time.time())); keys=list(row.keys())
        if not self._init or not os.path.exists(self.path):
            with open(self.path,'w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerow(row); self._init=True
        else:
            with open(self.path,'a',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writerow(row)
