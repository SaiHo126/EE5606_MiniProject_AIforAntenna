"""Complete the report's library-GP comparison without changing student sources.

Run from the repository root with the project's Python environment.
Uses GPR.py's exact model settings, the saved split and trained NN, and PSO.py.
Existing results are preserved in their original directories.
"""
from pathlib import Path
from time import perf_counter
import sys, json, warnings, hashlib, platform
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from NeuralNetwork import NeuralNetwork
from PSO import compare_models

OUT = ROOT / 'results' / 'report_library_gp'
OUT.mkdir(parents=True, exist_ok=True)

def metrics(actual, predicted):
    error = (np.asarray(predicted).ravel()-np.asarray(actual).ravel())*1000
    return dict(rmse_MHz=float(np.sqrt(np.mean(error**2))),
                mae_MHz=float(np.mean(abs(error))), max_error_MHz=float(abs(error).max()))

class LibraryPredictor:
    def __init__(self, scaler, gp):
        self.scaler, self.gp = scaler, gp
    def predict(self, x):
        return float(self.gp.predict(self.scaler.transform(np.asarray(x).reshape(1,3)))[0])

with np.load(ROOT / 'antennaSample_training.npz') as data:
    X, Y = data['X_mm'], data['Y_GHz'].ravel()
    ix = {s:data[s+'_indices'] for s in ['train','validation','test']}
scaler = StandardScaler().fit(X[ix['train']])
gp = GaussianProcessRegressor(kernel=RBF(length_scale=[1.,1.,1.]),
                              alpha=1e-8, normalize_y=True, random_state=1)
start = perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    gp.fit(scaler.transform(X[ix['train']]), Y[ix['train']])
fit_seconds=perf_counter()-start
predictor=LibraryPredictor(scaler,gp)
nn=NeuralNetwork.load(ROOT/'results/nn/nn_model_line_search.npz',cpu=True)
report={'gp_settings':'Exact settings in src/GPR.py; no test-set tuning',
        'fit_seconds':fit_seconds,'selected_kernel':str(gp.kernel_),
        'warnings':[str(w.message) for w in caught],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'data_seed':1,'split_seed':1,'test_samples':len(ix['test']),
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/'src/GPR.py',ROOT/'src/PSO.py',ROOT/'src/NeuralNetwork.py',ROOT/'antennaSample_training.npz']}}
for name,model in [('NN',nn),('GP',predictor)]:
    report[name]={s:metrics(Y[ids],[model.predict(x) for x in X[ids]]) for s,ids in ix.items()}
gp_pred=gp.predict(scaler.transform(X[ix['test']]))
nn_pred=np.array([nn.predict(x) for x in X[ix['test']]])
np.savez(OUT/'test_predictions.npz',indices=ix['test'],X_mm=X[ix['test']],actual_GHz=Y[ix['test']],NN_GHz=nn_pred,GP_GHz=gp_pred)
joblib.dump({'scaler':scaler,'gp':gp},OUT/'gp_model.joblib')
np.savez(OUT/'split_indices.npz',**ix)
compare_models(nn,predictor,1.,0.01,OUT)
report['timing_scope']='New CPU runs; fit timing excludes scaling, search excludes fitting/loading/final verification.'
(OUT/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
