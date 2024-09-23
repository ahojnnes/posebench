import h5py
import numpy as np
from utils.geometry import *
from utils.misc import *
import poselib
import datetime
import posebench
import collections
import cv2
from tqdm import tqdm

# Compute metrics for homography estimation
# AUC for max(err_R,err_t) and avg/med for runtime
def compute_metrics(results, thresholds = [5.0, 10.0, 20.0]):
    methods = results.keys()
    metrics = {}
    for m in methods:
        max_err = [a for (a,b) in results[m]['errs']]
        metrics[m] = {}
        aucs = compute_auc(max_err, thresholds)
        for auc, t in zip(aucs, thresholds):
            metrics[m][f'AUC{int(t)}'] = auc
        metrics[m]['avg_rt'] = np.mean(results[m]['runtime'])
        metrics[m]['med_rt'] = np.median(results[m]['runtime'])

    return metrics

def eval_homography_estimator(instance, estimator='poselib'):
    opt = instance['opt'].copy()
    threshold = instance['threshold']

    if estimator == 'poselib_approx':
        opt["use_approx_stopping"] = True
        tt1 = datetime.datetime.now()
        H, info = poselib.estimate_homography(instance['x1'], instance['x2'], opt, {})
        tt2 = datetime.datetime.now()
    elif estimator == 'poselib_exact':
        opt["use_approx_stopping"] = False
        tt1 = datetime.datetime.now()
        H, info = poselib.estimate_homography(instance['x1'], instance['x2'], opt, {})
        tt2 = datetime.datetime.now()
    elif estimator == 'pycolmap':
        opt = poselib_opt_to_pycolmap_opt(opt)
        tt1 = datetime.datetime.now()
        result = pycolmap.homography_matrix_estimation(instance['x1'], instance['x2'], opt)
        tt2 = datetime.datetime.now()
        H = result['H']

    K1 = camera_dict_to_calib_matrix(instance['cam1'])
    K2 = camera_dict_to_calib_matrix(instance['cam2'])
    Hnorm = np.linalg.inv(K2) @ H @ K1

    _, rotations, translations, _ = cv2.decomposeHomographyMat(Hnorm, np.identity(3))

    best_err_R = 180.0
    best_err_t = 180.0

    for k in range(len(rotations)):
        R = rotations[k]
        t = translations[k][:,0]

        err_R = rotation_angle(instance['R'] @ R.T)
        err_t = angle(instance['t'], t)

        if err_R < best_err_R:
            best_err_R = err_R
            best_err_t = err_t
    
    return [best_err_R, best_err_t], (tt2-tt1).total_seconds()

def main(dataset_path='data/homography', force_opt = {}, dataset_filter=[], method_filter = []):
    datasets = [
        ('barath_Alamo', 1.0),
        ('barath_NYC_Library', 1.0),
    ]
    if len(dataset_filter) > 0:
        datasets = [(n,t) for (n,t) in datasets if substr_in_list(n,dataset_filter)]

    evaluators = {
        'H (poselib, approx)': lambda i: eval_homography_estimator(i, estimator='poselib_approx'),
        'H (poselib, exact)': lambda i: eval_homography_estimator(i, estimator='poselib_exact'),
        # 'H (COLMAP)': lambda i: eval_homography_estimator(i, estimator='pycolmap'),
    }
    if len(method_filter) > 0:
        evaluators = {k:v for (k,v) in evaluators.items() if substr_in_list(k,method_filter)}

    metrics = {}
    full_results = {}
    for (dataset, threshold) in datasets:
        f = h5py.File(f'{dataset_path}/{dataset}.h5', 'r')

        results = collections.defaultdict(lambda: {"errs": [], "runtime": []})

        # RANSAC options
        opt = {
            'max_reproj_error': threshold,
            'max_epipolar_error': threshold,
            'max_iterations': 10000,
            'min_iterations': 1,
        }

        # Add in global overrides
        for k, v in force_opt.items():
            opt[k] = v

        # Since the datasets are so large we only take the first 1k pairs
        pairs = list(f.keys())
        if len(pairs) > 1000:
            pairs = pairs[0:1000]

        for k in tqdm(pairs, desc=dataset):
            v = f[k]
            full_num_samples = v['x1'][:].shape[0]

            np.random.seed(0)
            for num_samples in [20, 100, 200, 1000]:
                if full_num_samples < num_samples:
                    continue
                for success_prob in [0.95, 0.99]:
                    opt['success_prob'] = success_prob
                    for _ in range(10):  # Run on multiple permutations to reduce randomness.
                        instance = {
                            'x1': v['x1'][:],
                            'x2': v['x2'][:],
                            'cam1': h5_to_camera_dict(v['camera1']),
                            'cam2': h5_to_camera_dict(v['camera2']),
                            'R': v['R'][:],
                            't': v['t'][:],
                            'threshold': threshold,
                            'opt': opt
                        }
                        rand_idxs = np.random.choice(instance['x1'].shape[0], num_samples, replace=False)
                        instance['x1'] = instance['x1'][rand_idxs]
                        instance['x2'] = instance['x2'][rand_idxs]
                        for name, fcn in evaluators.items():
                            errs, runtime = fcn(instance)
                            results[name + f" [n={num_samples}, s={opt['success_prob']}]"]['errs'].append(np.array(errs))
                            results[name + f" [n={num_samples}, s={opt['success_prob']}]"]['runtime'].append(runtime)

        metrics[dataset] = compute_metrics(results)
        full_results[dataset] = results

    return metrics, full_results

if __name__ == '__main__':
    force_opt, method_filter, dataset_filter = posebench.parse_args()
    metrics, _ = main(force_opt=force_opt, method_filter=method_filter, dataset_filter=dataset_filter)
    posebench.print_metrics_per_dataset(metrics)
