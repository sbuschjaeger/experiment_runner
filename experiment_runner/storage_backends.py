import json
import os
import inspect
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from pymongo import MongoClient


class StorageBackend(ABC):
    @abstractmethod
    def write_experiment_config(self, cfg: dict):
        pass

    @abstractmethod
    def add_result(self, result: dict):
        pass

    def __json_encoder__(self, v):
        if isinstance(v, np.generic):
            return v.item()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, partial):
            return v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args]) + str(v.keywords)
        elif callable(v) or inspect.isclass(v):
            try:
                return v.__name__
            except:
                return str(v) #.__name__
        elif isinstance(v, object) and v.__class__.__module__ != 'builtins':
            # print(type(v))
            return str(v)
        else:
            raise TypeError()


class FSStorageBackend(StorageBackend):
    def __init__(self, out_path):
        self.out_path = os.path.abspath(out_path)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        else:
            if os.path.isfile(os.path.join(self.out_path, "results.jsonl")):
                os.unlink(os.path.join(self.out_path, "results.jsonl"))

    def write_experiment_config(self, cfg: dict):
        # Get experiment id from config.
        experiment_id = cfg["experiment_id"]

        # Construct output path.
        out_path = os.path.join(self.out_path, str(experiment_id))

        # Check, whether the path exists otherwise create the directories.
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Write the config to disk.
        with open(os.path.join(out_path, "config.json"), 'w') as out:
            out.write(json.dumps(cfg, indent=4, default=self.__json_encoder__))

    def add_result(self, result: dict):
        with open(os.path.join(self.out_path, "results.jsonl"), "a", 1) as out_file:
            out_file.write(json.dumps(result, sort_keys=True, default=self.__json_encoder__) + "\n")


class MongoDBStorageBackend(StorageBackend):
    def __init__(self, mongo_host: str, mongo_port: int, mongo_database: str):
        # Save properties.
        self.host = mongo_host
        self.port = mongo_port
        self.database = mongo_database

    def write_experiment_config(self, cfg: dict):
        client = MongoClient(self.host, self.port)
        client[self.database]["experiments"].insert_one(json.loads(json.dumps(cfg, default=self.__json_encoder__)))  # dumping + loading ensures, that no JSON-incompatible objects are saved to MongoDB

    def add_result(self, result: dict):
        client = MongoClient(self.host, self.port)
        client[self.database]["results"].insert_one(json.loads(json.dumps(result, default=self.__json_encoder__)))  # dumping + loading ensures, that no JSON-incompatible objects are saved to MongoDB
