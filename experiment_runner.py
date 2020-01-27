import multiprocessing
import multiprocessing.pool
from multiprocessing import Manager
import json
import os
import shutil
import time
import inspect
import traceback
import copy

import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from functools import partial

import numpy as np

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def replace_objects(d):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_objects(v)
        elif isinstance(v, partial):
            d[k] = v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args])
        elif callable(v) or inspect.isclass(v):
            try:
                d[k] = v.__name__
            except Exception as e:
                d[k] = str(v) #.__name__
    return d

def cfg_to_str(cfg):
    cfg = replace_objects(cfg.copy())
    return json.dumps(cfg, indent=4)

def store_model(out_path):
    # TODO IMPLEMENT
    pass

def eval_model(modelcfg, metrics, get_split, seed, experiment_id, run_id, out_path, result_file, store = False, verbose = True):
    lock.acquire()
    if cuda_devices_available is not None:
        cuda_device = cuda_devices_available.pop(0)
    else:
        cuda_device = None
    lock.release()

    # Make a copy of the model config for all output-related stuff
    # This does not include any fields which hurt the output (e.g. x_test,y_test)
    # but are usually part of the original modelcfg
    readable_modelcfg = copy.deepcopy(modelcfg)
    readable_modelcfg["verbose"] = verbose
    readable_modelcfg["seed"] = seed

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if verbose:
        print("STARTING EXPERIMENT {}-{} WITH CONFIG {}".format(experiment_id,run_id,cfg_to_str(readable_modelcfg)))
        print("\t {}-{} LOADING DATA".format(experiment_id, run_id))

    x_train, y_train, x_test,y_test = get_split(run_id = run_id)

    if verbose:
        print("\t {}-{} LOADING DATA DONE".format(experiment_id, run_id))
   

    # Prepare dict for model creation 
    model_ctor = modelcfg.pop("model")
    modelcfg["x_test"] = x_test
    modelcfg["y_test"] = y_test
    modelcfg["verbose"] = verbose
    modelcfg["seed"] = seed
    modelcfg["out_path"] = out_path

    model = model_ctor(**modelcfg)

    if cuda_device is not None:
        import torch
        with torch.cuda.device(cuda_device):
            start_time = time.time()
            model.fit(x_train, y_train)
            fit_time = time.time() - start_time

            scores = {}
            scores["fit_time"] = fit_time
            for name, fun in metrics.items():
                if x_train is not None and y_train is not None:
                    if verbose:
                        print("\t {}-{} SCORING {} ON TRAIN DATA".format(experiment_id, run_id, str(fun)))
                    scores[name + "_train"] = fun(model, x_train, y_train)

                if x_test is not None and y_test is not None:
                    if verbose:
                        print("\t {}-{} SCORING {} ON TEST DATA".format(experiment_id, run_id, str(fun)))
                    scores[name + "_test"] = fun(model, x_test, y_test)
    else:
        start_time = time.time()
        model.fit(x_train, y_train)
        fit_time = time.time() - start_time

        scores = {}
        scores["fit_time"] = fit_time
        for name, fun in metrics.items():
            if x_train is not None and y_train is not None:
                if verbose:
                    print("\t {}-{} SCORING {} ON TRAIN DATA".format(experiment_id, run_id, str(fun)))
                scores[name + "_train"] = fun(model, x_train, y_train)

            if x_test is not None and y_test is not None:
                if verbose:
                    print("\t {}-{} SCORING {} ON TEST DATA".format(experiment_id, run_id, str(fun)))
                scores[name + "_test"] = fun(model, x_test, y_test)
    
    readable_modelcfg["scores"] = scores
    if store:
        store_model(model, out_path)

        with open(out_path + "/config.json", 'w') as out:
            out.write(json.dumps(replace_objects(readable_modelcfg)))

    out_file = open(result_file,"a",1)
    lock.acquire()
    if cuda_devices_available is not None:
        cuda_devices_available.append(cuda_device)
    out_file.write(json.dumps(replace_objects(readable_modelcfg)) + "\n")
    lock.release()

    return experiment_id, run_id, scores
    
def get_train_test(basecfg, run_id):
    if "train" in basecfg and "test" in basecfg:
        x_train,y_train = basecfg["data_loader"](basecfg["train"])
        x_test,y_test = basecfg["data_loader"](basecfg["test"])
    else:
        from sklearn.model_selection import KFold
        X,y = basecfg["data_loader"](basecfg["data"])
        kf = KFold(n_splits=basecfg.get("no_runs", 1), random_state=basecfg.get("seed", None), shuffle=True)
        # TODO: This might be memory inefficient since list(..) materialises all splits, but only one is actually needed
        train_idx, test_idx = list(kf.split(X))[run_id]
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]
    
    return x_train,y_train,x_test,y_test

def run_experiments(basecfg, models, cuda_devices = None, n_cores = 8):
    try:
        return_str = ""
        results = []
        def init(l, cd_avail):
            global lock
            global cuda_devices_available
            
            lock = l
            cuda_devices_available = cd_avail

        if not os.path.exists(basecfg["out_path"]):
            os.makedirs(basecfg["out_path"])
        else:
            if os.path.isfile(basecfg["out_path"] + "/results.jsonl"):
                os.unlink(basecfg["out_path"] + "/results.jsonl")

        l = multiprocessing.Lock()
        
        manager = Manager()
        shared_list = manager.list(cuda_devices)
        no_gpus = 0 if not cuda_devices else len(set(cuda_devices))
        print("Starting {} experiments on {} cores using {} GPUs".format(len(models), n_cores, no_gpus))
        
        no_runs = basecfg.get("no_runs", 1)
        seed = basecfg.get("seed", None)

        experiments = []
        for experiment_id, modelcfg in enumerate(models):
            for run_id in range(no_runs):
                experiments.append(
                    (
                        modelcfg,
                        basecfg["scoring"],
                        partial(get_train_test,basecfg=basecfg),
                        seed,
                        experiment_id,
                        run_id,
                        basecfg.get("out_path", ".") + "/{}-{}".format(experiment_id,run_id),
                        basecfg["out_path"] + "/results.jsonl",
                        basecfg.get("store", False),
                        basecfg.get("verbose", False)
                    )
                )

        total_no_experiments = len(experiments)
        pool = MyPool(n_cores, initializer=init, initargs=(l,shared_list))
        for total_id, eval_return in enumerate(pool.starmap(eval_model, experiments)):
            experiment_id, run_id, results = eval_return
            accuracy = results.get("accuracy_test", 0)*100.0
            fit_time = results.get("fit_time", 0)
            print("{}/{} FINISHED. LAST EXPERIMENT WAS {}-{} WITH ACC {} in {} s".format(total_id+1, total_no_experiments,experiment_id,run_id,accuracy,fit_time))

        pool.close()
        pool.join()
    except Exception as e:
        return_str = str(e) + "\n"
        return_str += traceback.format_exc() + "\n"
    finally:
        print(return_str)
        if "mail" in basecfg:
            if "smtp_server" in basecfg:
                server = smtplib.SMTP(host=basecfg["smtp_server"],port=25)
            else:
                server = smtplib.SMTP(host='postamt.cs.uni-dortmund.de',port=25)

            msg = MIMEMultipart()
            msg["From"] = socket.gethostname()  + "@tu-dortmund.de"
            msg["To"] = basecfg["mail"]
            if "name" in basecfg:
                msg["Subject"] = "{} finished on {}".format(basecfg["name"], socket.gethostname())
            else:
                return_str = "All experiments finished on " + socket.gethostname()

            msg.attach(MIMEText(return_str, "plain"))
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        
