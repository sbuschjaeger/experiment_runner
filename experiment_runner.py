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

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# You have to scroll a bit to find this part.
class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess

        return proc

# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)

# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(multiprocessing.pool.Pool):
#     Process = NoDaemonProcess

def replace_objects(d):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_objects(v)
        elif isinstance(v, partial):
            # print(v)
            # print(v.args)
            # print(v.keywords)
            # asdf
            d[k] = v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args]) + str(replace_objects(v.keywords))
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

# getfullargspec does not handle inheritance correctly. 
# Taken from https://stackoverflow.com/questions/36994217/retrieving-arguments-from-a-class-with-multiple-inheritance
def get_ctor_arguments(clazz):
    args = ['self']
    for C in clazz.__mro__:
        if '__init__' in C.__dict__:
            args += inspect.getfullargspec(C).args[1:]
    return args

def eval_model(experiment_config):
    # TODO MAKE THIS NICER 
    # Unpack the whole config
    modelcfg = experiment_config[0]
    metrics = experiment_config[1]
    get_split = experiment_config[2]
    seed = experiment_config[3]
    experiment_id = experiment_config[4]
    no_runs = experiment_config[5]
    out_path = experiment_config[6]
    result_file = experiment_config[7]
    store = experiment_config[8]
    verbose = experiment_config[9]

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

    scores = {}
    scores["fit_time"] = []
    for m in metrics.keys():
        scores[m+ "_train"] = []
        scores[m+ "_test"] = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(out_path + "/config.json", 'w') as out:
        out.write(json.dumps(replace_objects(readable_modelcfg)))

    for run_id in range(no_runs):
        if verbose:
            print("STARTING EXPERIMENT {}-{} WITH CONFIG {}".format(experiment_id,run_id,cfg_to_str(readable_modelcfg)))
            print("\t {}-{} LOADING DATA".format(experiment_id, run_id))

        x_train, y_train, x_test,y_test = get_split(run_id = run_id)

        if verbose:
            print("\t {}-{} LOADING DATA DONE".format(experiment_id, run_id))

        # Prepare dict for model creation 
        tmpcfg = copy.deepcopy(modelcfg)
        model_ctor = tmpcfg.pop("model")
        if "x_test" not in tmpcfg and "y_test" not in tmpcfg:
            tmpcfg["x_test"] = x_test
            tmpcfg["y_test"] = y_test
        tmpcfg["verbose"] = verbose
        tmpcfg["seed"] = seed
        tmpcfg["out_path"] = out_path

        expected = {}
        for key in get_ctor_arguments(model_ctor):
            if key in tmpcfg:
                expected[key] = tmpcfg[key]

        model = model_ctor(**expected)

        if cuda_device is not None:
            import torch
            with torch.cuda.device(cuda_device):
                start_time = time.time()
                model.fit(x_train, y_train)
                fit_time = time.time() - start_time
                model.eval()
                scores["fit_time"].append(fit_time)
                for name, fun in metrics.items():
                    if x_train is not None and y_train is not None:
                        scores[name + "_train"].append(fun(model, x_train, y_train))
                    
                    if x_test is not None and y_test is not None:
                        scores[name + "_test"].append(fun(model, x_test, y_test))
        else:
            start_time = time.time()
            model.fit(x_train, y_train)
            fit_time = time.time() - start_time

            scores["fit_time"].append(fit_time)
            for name, fun in metrics.items():
                if x_train is not None and y_train is not None:
                    scores[name + "_train"].append(fun(model, x_train, y_train))

                if x_test is not None and y_test is not None:
                    scores[name + "_test"].append(fun(model, x_test, y_test))

        if store:
            print("STORING")
            # TODO ADD RUN_ID to path
            store_model(model, out_path)

    readable_modelcfg["scores"] = scores
    out_file = open(result_file,"a",1)
    lock.acquire()
    if cuda_devices_available is not None:
        cuda_devices_available.append(cuda_device)
    out_file.write(json.dumps(replace_objects(readable_modelcfg), sort_keys=True) + "\n")
    lock.release()

    print("DONE")
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
        if cuda_devices is None or len(cuda_devices) == 0:
            no_gpus = 0
            shared_list = None
        else:
            shared_list = manager.list(cuda_devices)
            no_gpus = len(set(cuda_devices))
        print("Starting {} experiments on {} cores using {} GPUs".format(len(models), n_cores, no_gpus))
        
        no_runs = basecfg.get("no_runs", 1)
        seed = basecfg.get("seed", None)

        experiments = []
        for experiment_id, modelcfg in enumerate(models):
            experiments.append(
                (
                    modelcfg,
                    basecfg["scoring"],
                    partial(get_train_test,basecfg=basecfg),
                    seed,
                    experiment_id,
                    no_runs,
                    basecfg.get("out_path", ".") + "/{}".format(experiment_id),
                    basecfg["out_path"] + "/results.jsonl",
                    basecfg.get("store", False),
                    basecfg.get("verbose", False)
                )
            )

        total_no_experiments = len(experiments)
        pool = NonDaemonPool(n_cores, initializer=init, initargs=(l,shared_list))
        # Lets use imap and not starmap to keep track of the progress
        for total_id, eval_return in enumerate(pool.imap_unordered(eval_model, experiments)):
            experiment_id, run_id, results = eval_return
            accuracy = results.get("accuracy_test", 0)
            fit_time = results.get("fit_time", 0)
            print("{}/{} FINISHED. LAST EXPERIMENT WAS {}-{} WITH ACC {} in {} s".format(total_id+1, total_no_experiments,experiment_id,run_id,np.mean(accuracy)*100.0,np.mean(fit_time)))

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
        
