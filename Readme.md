# Experiment Runner

This is a simple package / script which can be used to run multiple experiments in parallel on the same machine or distributed across many different machines. This script makes minimal assumptions about the kind of experiments you want to run and focuses on flexibility. The core idea is that there are three functions `pre` -> `fit` -> `post` for each experiment:

- `pre(cfg)`: Is called before the experiment defined by `cfg` is performed. You can place anything here what you want to do before starting the experiment, e.g. loading the data or creating a model. Anything which is returned by this function is passed to the next call. If you don't return anything, just return `None`. 

- `fit(cfg, returned_by_pre)`: Is called after `pre` has been called. Whatever has been returned by `pre` is passed to this function through `returned_by_pre`. Place any code for the actual experiment in this function. For example, if you want to fit a model you should place the code here. Make sure, that you return anything (also stuff computed by `pre`) which you might need in the next function. If you don't return anything, just return `None`. 

- `post(cfg, returned_by_fit)`: Is called after `fit` has been called. Whatever has been returned by `fit` is passed to this function through `returned_by_fit`. Usually you want to compute some statistics / metrics about your experiment, e.g. a test error, test loss etc. This function should return a dictionary with key/value pairs for each metrics. If you do not compute any metrics, return an empty dictionary `{}`. Note that the `experiment_runner` will automatically add a field `fit_time` into this dictionary which is the time spend in the `fit` function (measured via `time.time()`). 

Each experiment is defined by a basis config `basecfg` and a list of individual configs `cfgs`. The `basecfg` is a dictionary containing the basis configuration of the experiment with the following fields:

- `out_path`: The path in which results should be written to. All scores returned by `post` will be gathered and stored under `${out_path}/results.jsonl`. Moreover, each individual configuration will be stored under `${out_path}/$id/$rep/config.json` where `$id` is a unique identifier for the experiment (starting by 0) and `$rep` is the current repetition of the experiment (see below). If `n_repeitions` is 1, then `$rep` is omitted an the path becomes `${out_path}/$id/config.json`. Note that `pre`/`fit`/`post` will receive the adjusted `out_path` which contains the `$id` and `$rep` (if any). Also note, that the individual configs `cfgs` should be serializable to JSON files so that the config can be properly stored. The `experiment_runner` will attempt to convert list / numpy arrays accordingly, but I would not rely on this code for reproducibility. See the "best practices" section below for more information on that. 
- `pre` (required): The pre function.
- `fit` (required): The fit function.
- `post` (required): The post function.
- `backend` (optional, defaults to `single`): The backend you want to use for running the experiments. Currently supported are {`multiprocessing`, `ray`, `single`, `malocher`}. Any string which is not `multiprocessing`, `malocher` or `ray` will be interpreted as `single`. As the names suggest, `multiprocessing` will run experiment on the same machine using multiple processes (using a `multiprocessing.Pool`). `ray` uses [Ray](https://github.com/ray-project/ray) to distribute experiments across multiple machines, `malocher`uses [malocher](htts://github.com/Whadup/malocher) to run the experiments on multiple ssh-machines and `single` just runs one experiment after another on the current machine without multi-threading.
- `num_cpus` (optional, only used by {`multiprocessing`, `ray`}, defaults to 1): The number of cpus / threads used by each experiment
- `num_gpus` (optional, only used by {`ray`}, defaults to 0): The number of gpus required by each experiment
- `max_memory` (optional, only used by {`ray`}, defaults to 1GB): The maximum number of memory required by the experiment. 
- `address` (optional, only used by {`ray`}, defaults to `auto`): Address of the ray head
- `redis_password` (optional, only used by {`ray`}, defaults to `None`): Redis password of the ray head
- `verbose` (optional {`True`, `False`}, defaults to `True`): Displays a TQDM progress bar over all experiments.
- `repetitions` (optional, defaults to 1): How often this experiment should be repeated (e.g. due to randomness). Note that the `experiment_runner` does not take care of random seeds, but you have to implement it in `fit`.
- `timeout` (optional, defaults to 0): Sets an optional timeout in seconds. A single experiment is stopped after `timeout` seconds. No statistcs are kept and an exception is printed. Execution of other experiments is resumed as usual.
  

An example `basecfg` could be:

    basecfg = {
        "out_path":"results/",
        "pre": pre,
        "post": post,
        "fit": fit,
        "backend": "multiprocessing",
        "num_cpus":8,
        "verbose":True
    }


The list of individual configurations `cfgs` can be pretty much whatever you want. The following keywords are reserved however:

- `experiment_id`: The unique id of the experiment starting by 0.
- `run_id`: The current repetition of the experiment starting by 0.
- `out_path`: The corresponding file path as detailed above.

An example `experiment_cfg` could be:

    cfg = {
        # Whatever you need for your experiment
    }
    
    experiment_cfg = {
        **cfg, 
        'experiment_id':experiment_id,
        'out_path':rep_out_path, 
        'run_id':i
    }

Similarly, you can return whatever scores you need for your experiment. The following keywords are reserved however:

- `fit_time`: The amount of time spend in the `fit` function measured via `time.time()`. This field is automatically added by `experiment_runner`
- `mean_$M`: If `n_repetitions` is greater than 1, for each metric `$M` the mean over all runs is added
- `std_$M`: If `n_repetitions` is greater than 1, for each metric `$M` the standard deviation over all runs is added



## Hyperparameter search

The `experiment_runner` has some basic support for hyperparameter search. We follow the philosophy discussed in "Random Search for Hyper-Parameter Optimization" by Bergstra and Bengio, JMLR (13) in 2012 (https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf) which states that randomly chosen hyperparameter often outperform a classic grid search. In order to perform hyperparameter search you can tag a parameter as `Variation` which receives a list of possible variations for the hyperparameter and then call `generate_configs` with the desired number of configurations you want to check. For example you can generate `n_configs = 5` different hyperparameter configurations for two parameters: 

    generate_configs(
        cfg = {
            "param_1": Variation([1,2,3,4]),
            "param_2": Variation([1e-1,1e-2,1e-3]),
        },
        n_configs = 3
    )

`generate_configs` will recursively check all dictionary for `Variation` classes and return a list with `n_configs` configurations. Every occurrence of `Variation` is replaced by a randomly chosen entry from the supplied list. The function takes care of duplicate entries and makes sure that only unique hyperparameter configurations are returned. It also checks if there are at-least `n_configs` configurations possible and adapts `n_configs` accordingly. The result might look like:

    [
        {
            "param_1": 1,
            "param_2": 1e-1,
        },
        {
            "param_1": 2,
            "param_2": 1e-3,
        },
        {
            "param_1": 4,
            "param_2": 1e-1,
        }
    ]

## Accessing results

The results are written to a json-line file which can be read via the following snippet:

    import json 
    from pandas.io.json import json_normalize 
    
    def read_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    
        return json_normalize(data)

The `json_normalize` call will return a pandas data frame which is flattend and thereby column names change a bit. For example, you have to access the `fit_time` via `scores.fit_time` and so on.

## Best practices

- One of the central ideas of the `experiment_runner` is, that a dictionary defines a single experiment and that this dictionary can be stored in a json file which can be read by a human and reproduced by a machine. Python objects and functions are difficult to store and thus it is best practice to configure experiments stuff which can be easily stored in a json file, e.g. strings, lists or numbers. The main purpose of the `pre` function is then to extract these strings and create the corresponding objects. This can be a bit tedious form time to time, but the following snippets might help:

    ### Find a python class given the corresponding classname
    
        import sys
        
        def str_to_class(classname):
            return getattr(sys.modules[__name__], classname)
    
    ### Find a python function given the corresponding function name
    
        import foo
        method_to_call = getattr(foo, 'bar')
        result = method_to_call()

    ### Brute-force method
    
        eval("Foo")

- When running many experiments which differ in runtime it is a good idea to shuffle them beforehand so that the get randomly distributed across ray / multiprocessing:

        basecfg = {...}
        experiments = [...]
        random.shuffle(experiments)
        run_experiments(basecfg, experiments)

- You should somehow organize the results, e.g. by using the current time + date for the  `out_path`:

        from datetime import datetime
        basecfg = {
            "out_path":"results/" + datetime.now().strftime('%d-%m-%Y-%H:%M:%S'),
            "pre": pre,
            "post": post,
            "fit": fit,
            "backend": "single",
            "verbose":True
        }

- The `results.jsonl` contains a lot of information which is usually not required for displaying results. Moreover, entries are often close to programming and not close to what you would write in a paper. The following snippets can be helpful

    ### Nicer column names and rounding
        def nice_name(row):
            # Build a nice looking string
            return "{} {}".format(row["model"], row["param_1"])
        df = read_jsonl(os.path.join(latest_folder, "results.jsonl"))
        df["nice_name"] = df.apply(nice_name, axis=1)
        df = df.round(decimals = 3)

    ### Select relevant columns and sort by mean_accuracy
        tabledf = df[["nice_name", "mean_accuracy", "mean_params", "scores.mean_fit_time"]]
        tabledf = tabledf.sort_values(by=['mean_accuracy'], ascending = False)
        display(HTML(shortdf.to_html()))

    ### Select the best method according to mean_accuracy
        idx = tabledf.groupby(['nice_name'])['mean_accuracy'].transform(max) == tabledf['mean_accuracy']
        shortdf = tabledf[idx]
        display(HTML(shortdf.to_html()))
