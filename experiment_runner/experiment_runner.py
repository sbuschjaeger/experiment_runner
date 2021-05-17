"""Experiment Runner. It's great!"""
from functools import partial
import json
import os
import inspect
import random
import time
import traceback
from multiprocessing import Pool
import copy
import signal
import numpy as np
from tqdm import tqdm
try:
    import ray
    @ray.remote(max_calls=1)
    def ray_eval_fit(pre, fit, post, out_path, experiment_id, cfg):
        """wraps eval_fit to run with ray .remote()"""
        return eval_fit((pre, fit, post, out_path, experiment_id, cfg))
except ImportError as error:
    ray = None
try:
    import malocher
except ImportError as error:
    malocher = None

def stacktrace(exception):
    """convenience method for java-style stack trace error messages"""
    print("\n".join(traceback.format_exception(None, exception, exception.__traceback__)),
        #file=sys.stderr,
        flush=True)

def replace_objects(d):
    """
    convenience method to json-serialize configs that include objects,
    partials, numpy-arrays and other non-json-conform things.
    """
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_objects(v)
        elif isinstance(v, list):
            d[k] = [replace_objects({"key":vv})["key"] for vv in v]
        elif isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, np.ndarray):
            d[k] = k
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

def get_ctor_arguments(clazz):
    """
    getfullargspec does not handle inheritance correctly.
    Taken from https://stackoverflow.com/questions/36994217/retrieving-arguments-from-a-class-with-multiple-inheritance
    """
    args = ['self']
    for C in clazz.__mro__:
        if '__init__' in C.__dict__:
            args += inspect.getfullargspec(C).args[1:]
            args += inspect.getfullargspec(C).kwonlyargs
    return args

class Variation:
    """Handles testing different hyperparameter variantions"""
    def __init__(self, list_of_choices):
        self.choices = list_of_choices

    def get(self):
        """return a random variant"""
        return np.random.choice(self.choices)


def generate_configs(cfg, n_configs):
    """Resolve configs which include `Variation` parameters"""
    configs = []

    def n_variations(d):
        n_choices = []
        for val in d.values():
            if isinstance(val, Variation):
                n_choices.append(len(val.choices))
            elif isinstance(val, dict):
                n_choices.append(n_variations(val))
        return np.prod(n_choices)

    def vary_dict(d):
        new_dict = {}
        for key, val in d.items():
            if isinstance(val, Variation):
                new_dict[key] = val.get()
            elif isinstance(val, dict):
                new_dict[key] = vary_dict(val)
            else:
                new_dict[key] = val

        return new_dict

    possible_variations = n_variations(cfg)
    if possible_variations < n_configs:
        n_configs = possible_variations

    while len(configs) < n_configs:
        new_config = vary_dict(cfg)
        if new_config not in configs:
            configs.append(new_config)

    return configs

def raise_timeout(signum, frame):
    """because lambdas cannot raise exceptions?"""
    raise TimeoutError()

def eval_fit(config):
    """
    Central internal method of the that calls pre, fit and post,
    handles repeated execution of experiments, measures fit time and
    stores results.
    """
    pre, fit, post, timeout, out_path, experiment_id, cfg = config
    if timeout > 0:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.alarm(timeout)
    try:
        # Make a copy of the model config for all output-related stuff
        # This does not include any fields which hurt the output (e.g. x_test,y_test)
        # but are usually part of the original modelcfg
        # if not verbose:
        #     import warnings
        #     warnings.filterwarnings('ignore')
        readable_cfg = copy.deepcopy(cfg)
        readable_cfg["experiment_id"] = experiment_id
        readable_cfg["out_path"] = out_path

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(os.path.join(out_path,  "config.json"), 'w') as out:
            out.write(json.dumps(replace_objects(readable_cfg), indent=4))

        scores = {}
        repetitions = cfg.get("repetitions", 1)
        for i in range(repetitions):
            if repetitions > 1:
                rep_out_path = os.path.join(out_path, str(i))
                if not os.path.exists(rep_out_path):
                    os.makedirs(rep_out_path)
            else:
                rep_out_path = out_path

            experiment_cfg = {
                **cfg,
                'experiment_id':experiment_id,
                'out_path':rep_out_path,
                'run_id':i
            }

            if pre is not None:
                pre_stuff = pre(experiment_cfg)
                start_time = time.time()
                fit_stuff = fit(experiment_cfg, pre_stuff)
                fit_time = time.time() - start_time
            else:
                start_time = time.time()
                fit_stuff = fit(experiment_cfg)
                fit_time = time.time() - start_time

            if post is not None:
                cur_scores = post(experiment_cfg, fit_stuff)
                cur_scores["fit_time"] = fit_time

                if i == 0:
                    for k in list(cur_scores.keys()):
                        scores[k] = [cur_scores[k]]
                else:
                    for k in list(scores.keys()):
                        scores[k].append(cur_scores[k])

        for k in list(scores.keys()):
            scores["mean_" + k] = np.mean(scores[k])
            scores["std_" + k] = np.std(scores[k])

        readable_cfg["scores"] = scores
        out_file_content = json.dumps(replace_objects(readable_cfg), sort_keys=True) + "\n"

        signal.alarm(0)
        return experiment_id, scores, out_file_content
    except Exception as identifier:
        stacktrace(identifier)
        # Ray is somtimes a little bit to quick in killing our processes if something bad happens
        # In this case we do not see the stack trace which is super annyoing. Therefore, we sleep a
        # second to wait until the print has been processed / flushed
        signal.alarm(0)
        time.sleep(1.0)
        return None



def run_experiments(basecfg, cfgs, **kwargs):
    """
    The main API call of the experiment_runner.
    Pass a base_cfg to configure the execution of the experiments.
    Pass a list of `cfgs` to specify each experiment.

    See readme for available basecfg settings and reserved cfg keys.
    """
    try:
        return_str = ""
        # results = []
        if "out_path" in basecfg:
            basecfg["out_path"] = os.path.abspath(basecfg["out_path"])

        if not os.path.exists(basecfg["out_path"]):
            os.makedirs(basecfg["out_path"])
        else:
            if os.path.isfile(basecfg["out_path"] + "/results.jsonl"):
                os.unlink(basecfg["out_path"] + "/results.jsonl")

        # pool = NonDaemonPool(n_cores, initializer=init, initargs=(l,shared_list))
        # Lets use imap and not starmap to keep track of the progress
        # ray.init(address="ls8ws013:6379")
        backend = basecfg.get("backend", "single")
        verbose = basecfg.get("verbose", True)

        print("Starting {} experiments via {} backend".format(len(cfgs), backend))

        if backend == "ray":
            ray.init(
                address=basecfg.get("address", "auto"),
                _redis_password=basecfg.get("redis_password", None)
            )

        if backend == "ray":
            configurations = [ray_eval_fit.options(
                        num_cpus=basecfg.get("num_cpus", 1),
                        num_gpus=basecfg.get("num_gpus", 0),
                        memory=basecfg.get("max_memory", 1000 * 1024 * 1024) # 1 GB
                    ).remote(
                        basecfg.get("pre", None),
                        basecfg.get("fit", None),
                        basecfg.get("post", None),
                        basecfg.get("timeout", 0),
                        os.path.join(basecfg["out_path"], str(experiment_id)),
                        experiment_id,
                        cfg
                    ) for experiment_id, cfg in enumerate(cfgs)
            ]
            print("SUBMITTED JOBS, NOW WAITING")
        else:
            configurations = [
                    (
                    basecfg.get("pre", None),
                    basecfg.get("fit", None),
                    basecfg.get("post", None),
                    basecfg.get("timeout", 0),
                    os.path.join(basecfg["out_path"], str(experiment_id)),
                    experiment_id,
                    cfg
                ) for experiment_id, cfg in enumerate(cfgs)
            ]

        if backend == "ray":
            # https://github.com/ray-project/ray/issues/8164
            def to_iterator(configs):
                while configs:
                    result, configs = ray.wait(configs)
                    yield ray.get(result[0])

            random.shuffle(configurations)
            for result in tqdm(to_iterator(configurations), total=len(configurations)):
                if result is not None:
                    experiment_id, results, out_file_content = result
                    with open(os.path.join(basecfg["out_path"], "results.jsonl"), "a", 1) as out_file:
                        out_file.write(out_file_content)
        elif backend == "malocher":
            malocher_dir = basecfg.get("malocher_dir", ".malocher_dir")
            malocher_machines = basecfg["malocher_machines"]
            malocher_user = basecfg["malocher_user"]
            malocher_port = basecfg.get("malocher_port", 22)
            malocher_key = basecfg.get("malocher_key", "~/.ssh/id_rsa")
            for cfg in configurations:
                malocher.submit(eval_fit, cfg, malocher_dir=malocher_dir)
            results = malocher.process_all(
                malocher_dir=malocher_dir,
                ssh_machines=malocher_machines,
                ssh_username=malocher_user,
                ssh_port=malocher_port,
                ssh_private_key=malocher_key,
            )
            for job_id, eval_return in tqdm(results, total=len(configurations), disable=not verbose):
                if eval_return is not None:
                    experiment_id, results, out_file_content = eval_return
                    with open(os.path.join(basecfg["out_path"], "results.jsonl"), "a", 1) as out_file:
                        out_file.write(out_file_content)


        elif backend == "multiprocessing":
            pool = Pool(basecfg.get("num_cpus", 1))
            for eval_return in tqdm(pool.imap_unordered(eval_fit, configurations), total=len(configurations), disable=not verbose):
                if eval_return is not None:
                    experiment_id, results, out_file_content = eval_return
                    with open(os.path.join(basecfg["out_path"], "results.jsonl"), "a", 1) as out_file:
                        out_file.write(out_file_content)
        else:
            for f in tqdm(configurations, disable=not verbose):
                eval_return = eval_fit(f)
                if eval_return is not None:
                    experiment_id, results, out_file_content = eval_return
                    with open(os.path.join(basecfg["out_path"], "results.jsonl"), "a", 1) as out_file:
                        out_file.write(out_file_content)
    except Exception as e:
        return_str = str(e) + "\n"
        return_str += traceback.format_exc() + "\n"
    finally:
        print(return_str)
        if backend == "ray":
            ray.shutdown()
