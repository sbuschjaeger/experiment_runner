import copy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import partial
import inspect
import json
import os
import smtplib
import socket
import time
import traceback
import joblib

from sklearn.pipeline import Pipeline 

import numpy as np
import ray

# TODO Change function header to be nicer
def store_model(previous_output, model, out_path = None, x_train = None, y_train = None, x_test = None, y_test = None):
    if hasattr(model, "store"):
        print("FOUND CUSTOM MODEL")
        # Custom model
        # TODO Remove dim options here
        model.store("{}".format(out_path), dim=x_train[0].shape, name="model") 
    else:
        if isinstance(model, Pipeline) and hasattr(model.steps[-1], "store"):
            print("FOUND PIPELINE WITH CUSTOM MODEL")
            # SKLEARN PIPELINE With custom model
            custom_model = model.steps.pop()
            joblib.dump(model, "{}model_pipeline.pkl".format(out_path))

            # TODO Remove dim options here
            custom_model.store("{}".format(out_path), dim=x_train[0].shape, name="model")
        else:
            print("FOUND PIPELINE")
            joblib.dump(model, "{}/model.pkl".format(out_path))
            # SKLEARN MODEL OR PIPELINE

@ray.remote
def eval_model(modelcfg, metrics, get_split, seed, experiment_id, no_runs, out_path, verbose):
    def replace_objects(d):
        d = d.copy()
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = replace_objects(v)
            elif isinstance(v, list):
                d[k] = [replace_objects({"key":vv})["key"] for vv in v]
            elif isinstance(v, np.generic):
                d[k] = v.item()
            elif isinstance(v, partial):
                d[k] = v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args]) + str(replace_objects(v.keywords))
            elif callable(v) or inspect.isclass(v):
                try:
                    d[k] = v.__name__
                except:
                    d[k] = str(v) #.__name__
            elif isinstance(v, object) and v.__class__.__module__ != 'builtins':
                # print(type(v))
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def stacktrace(exception):
        """convenience method for java-style stack trace error messages"""
        import sys
        import traceback
        print("\n".join(traceback.format_exception(None, exception, exception.__traceback__)),
            #file=sys.stderr,
            flush=True)

    def cfg_to_str(cfg):
        cfg = replace_objects(cfg.copy())
        return json.dumps(cfg, indent=4)

    # getfullargspec does not handle inheritance correctly.
    # Taken from https://stackoverflow.com/questions/36994217/retrieving-arguments-from-a-class-with-multiple-inheritance
    def get_ctor_arguments(clazz):
        args = ['self']
        for C in clazz.__mro__:
            if '__init__' in C.__dict__:
                args += inspect.getfullargspec(C).args[1:]
        return args
    try:
        # Make a copy of the model config for all output-related stuff
        # This does not include any fields which hurt the output (e.g. x_test,y_test)
        # but are usually part of the original modelcfg
        if not verbose:
            import warnings
            warnings.filterwarnings('ignore')
        readable_modelcfg = copy.deepcopy(modelcfg)
        readable_modelcfg["verbose"] = verbose
        readable_modelcfg["seed"] = seed
        readable_modelcfg["experiment_id"] = experiment_id

        scores = {}
        scores["fit_time"] = []
        for m in metrics.keys():
            scores[m+ "_train"] = []
            scores[m+ "_test"] = []

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(out_path + "/config.json", 'w') as out:
            out.write(json.dumps(replace_objects(readable_modelcfg), indent=4))

        for run_id in range(no_runs):
            if verbose:
                print("STARTING EXPERIMENT {}-{} WITH CONFIG {}".format(
                    experiment_id,
                    run_id,
                    cfg_to_str(readable_modelcfg)))
                print("\t {}-{} LOADING DATA".format(experiment_id, run_id))

            x_train, y_train, x_test, y_test = get_split(run_id=run_id)

            if verbose:
                print("\t {}-{} LOADING DATA DONE".format(experiment_id, run_id))

            # Prepare dict for model creation
            tmpcfg = copy.deepcopy(modelcfg)
            model_ctor = tmpcfg.pop("model")

            print("WARNING: Reloading ", model_ctor, "from Hard Disk. " + \
                  "If you changed the code since starting this experiment, " + \
                  "then you fucked up your experiment now.")
            # HACK: This should actually happen only once per Ray-Slave
            from importlib import reload, import_module
            # Make sure to use the most up-to-date code.
            module = reload(import_module(model_ctor.__module__))
            model_ctor = module.__getattribute__(model_ctor.__name__)
            if "x_test" not in tmpcfg and "y_test" not in tmpcfg:
                tmpcfg["x_test"] = x_test
                tmpcfg["y_test"] = y_test
            tmpcfg["verbose"] = verbose
            tmpcfg["seed"] = seed
            tmpcfg["out_path"] = out_path

            pre_pipeline = modelcfg.get("pre_pipeline", None)
            if pre_pipeline:
                from sklearn.base import clone
                from sklearn.pipeline import make_pipeline
                # print(pipeline)
                pre_pipeline = make_pipeline(*[clone(p) for p in pre_pipeline], "passthrough")
            tmpcfg["pipeline"] = pre_pipeline

            expected = {}
            for key in get_ctor_arguments(model_ctor):
                if key in tmpcfg:
                    expected[key] = tmpcfg[key]

            model = model_ctor(**expected)
            if pre_pipeline and "pipeline" not in expected:
                # Model cannot handle the pipeline internally,
                # so we put the model at the end of the pipeline and
                # train the whole pipeline instead of the model.
                pre_pipeline.steps[-1] = (model.__class__.__name__, model)
                model = pre_pipeline

            start_time = time.time()
            model.fit(x_train, y_train)
            fit_time = time.time() - start_time

            # TODO MAYBE ADD METRICS TO POST_PIPELINE?
            scores["fit_time"].append(fit_time)
            for name, fun in metrics.items():
                if x_train is not None and y_train is not None:
                    scores[name + "_train"].append(fun(model, x_train, y_train))

                if x_test is not None and y_test is not None:
                    scores[name + "_test"].append(fun(model, x_test, y_test))

            post_pipeline = modelcfg.get("post_pipeline", None)
            if post_pipeline is None:
                post_pipeline = []
            
            if verbose:
                print("RUNNING POST_PIPELINE")
            
            previous_output = None
            for step in post_pipeline:
                if verbose:
                    print("POST_PIPELINE STEP {}".format(step.__name__))
                
                previous_output = step(
                    previous_output, 
                    model, 
                    out_path, 
                    x_train, 
                    y_train, 
                    x_test, 
                    y_test
                )
            
        for score in list(scores.keys()):
            if len(scores[score]) > 1:
                scores["mean_"+ score] = np.array(scores[score]).mean()
                scores["std_"+ score] = np.array(scores[score]).std()
        readable_modelcfg["scores"] = scores

        out_file_content = json.dumps(replace_objects(readable_modelcfg), sort_keys=True) + "\n"

        print("DONE")
        return experiment_id, run_id, scores, out_file_content
    except Exception as identifier:
        stacktrace(identifier)
        # Ray is somtimes a little bit to quick in killing our processes if something bad happens 
        # In this case we do not see the stack trace which is super annyoing. Therefore, we sleep a
        # second to wait until the print has been processed / flushed
        time.sleep(1.0)
        return None

def run_experiments(basecfg, models, **kwargs):
    def get_train_test(basecfg, run_id):
        if "train" in basecfg and "test" in basecfg:
            x_train, y_train = basecfg["data_loader"](basecfg["train"])
            x_test, y_test = basecfg["data_loader"](basecfg["test"])
        else:
            from sklearn.model_selection import StratifiedKFold
            X, y = basecfg["data_loader"](basecfg["data"])
            xval = StratifiedKFold(n_splits=basecfg.get("no_runs", 1),
                       random_state=basecfg.get("seed", None),
                       shuffle=True)
            # TODO: This might be memory inefficient since list(..)
            # materialises all splits, but only one is actually needed
            train_idx, test_idx = list(xval.split(X, y))[run_id]
            x_train, y_train = X[train_idx], y[train_idx]
            x_test, y_test = X[test_idx], y[test_idx]
        return x_train, y_train, x_test, y_test

    try:
        return_str = ""
        results = []
        if "out_path" in basecfg:
            basecfg["out_path"] = os.path.abspath(basecfg["out_path"])

        if not os.path.exists(basecfg["out_path"]):
            os.makedirs(basecfg["out_path"])
        else:
            if os.path.isfile(basecfg["out_path"] + "/results.jsonl"):
                os.unlink(basecfg["out_path"] + "/results.jsonl")

        print("Starting {} experiments on Ray".format(len(models)))

        no_runs = basecfg.get("no_runs", 1)
        seed = basecfg.get("seed", None)

        # pool = NonDaemonPool(n_cores, initializer=init, initargs=(l,shared_list))
        # Lets use imap and not starmap to keep track of the progress
        # ray.init(address="ls8ws013:6379")
        ray.init(address=basecfg.get("ray_head", None), local_mode=basecfg.get("ray_local_mode", False))
        # ray.init(address=basecfg.get("ray_head", None), local_mode=False)
        
        for model_cfg in models:
            for cfg in basecfg:
                if cfg not in model_cfg:
                    model_cfg[cfg] = basecfg[cfg]

        futures = [eval_model.options(
                num_cpus=modelcfg.get("num_cpus", 1),
                num_gpus=modelcfg.get("num_gpus", 1)).remote(
                modelcfg,
                modelcfg["scoring"],
                partial(get_train_test, basecfg=modelcfg),
                seed,
                experiment_id,
                no_runs,
                modelcfg.get("out_path", ".") + "/{}".format(experiment_id),
                modelcfg.get("verbose", False)
            ) for experiment_id, modelcfg in enumerate(models)
        ]
        total_no_experiments = len(futures)
        total_id = 0
        while futures:
            result, futures = ray.wait(futures)
            eval_return = ray.get(result[0])
            if eval_return is None:
                print("NONE!")
                continue
            # for total_id, eval_return in enumerate(pool.imap_unordered(eval_model, experiments)):
            experiment_id, run_id, results, out_file_content = eval_return
            with open(basecfg["out_path"] + "/results.jsonl", "a", 1) as out_file:# HACK AROUND THIS
                out_file.write(out_file_content)

            accuracy = results.get("accuracy_test", 0)
            fit_time = results.get("fit_time", 0)
            print("{}/{} FINISHED. LAST EXPERIMENT WAS {}-{} WITH ACC {} in {} s".format(
                total_id+1, total_no_experiments, experiment_id, run_id,
                np.mean(accuracy) * 100.0, np.mean(fit_time)))
            total_id += 1
    except Exception as e:
        return_str = str(e) + "\n"
        return_str += traceback.format_exc() + "\n"
    finally:
        print(return_str)
        ray.shutdown()
        if "mail" in basecfg:
            if "smtp_server" in basecfg:
                server = smtplib.SMTP(host=basecfg["smtp_server"], port=25)
            else:
                server = smtplib.SMTP(host='postamt.cs.uni-dortmund.de', port=25)

            msg = MIMEMultipart()
            msg["From"] = socket.gethostname() + "@tu-dortmund.de"
            msg["To"] = basecfg["mail"]
            if "name" in basecfg:
                msg["Subject"] = "{} finished on {}".format(basecfg["name"], socket.gethostname())
            else:
                return_str = "All experiments finished on " + socket.gethostname()

            msg.attach(MIMEText(return_str, "plain"))
            server.sendmail(msg['From'], msg['To'], msg.as_string())
