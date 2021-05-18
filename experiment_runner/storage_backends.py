import json
import os
import inspect
import numpy as np
from functools import partial
from abc import ABC, abstractmethod
from pymongo import MongoClient


class StorageBackend(ABC):
    """
    Abstract base class (ABC) for storage backends.
    """
    @abstractmethod
    def write_experiment_config(self, cfg: dict):
        pass

    @abstractmethod
    def add_result(self, result: dict):
        pass

    def __json_encoder__(self, v):
        '''
        Custom JSON encoder, which may be used together with JSON.dump(s) + default. Commonly used for storing results through serialization.
        Handles some of the types, which are usually used within experimental routines.
        :param v: Object to serialize.
        :return: JSON-friendly object, which can be serialized using the json package.
        '''
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
    def __init__(self, out_path, force=False):
        """
        Initializes a storage backend which uses the local filesystem to store experiments and their results.

        :param out_path: Directory path, which will be used to store experiments and result.
        :param force: Allows initialization of the storage backend even when the output path is existent and not empty.
        """
        self.out_path = os.path.abspath(out_path)

        # Check, whether initialization should be refused.
        if not force and os.path.exists(self.out_path) and os.listdir(self.out_path):
            raise ValueError(f"FSStorageBackend: Output directory at '{self.out_path}' does already exist and is not empty.")

        # Create necessary directory structures.
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        else:
            if os.path.isfile(os.path.join(self.out_path, "results.jsonl")):
                os.unlink(os.path.join(self.out_path, "results.jsonl"))

    def write_experiment_config(self, cfg: dict):
        """
        Writes the configuration of an experiment to the database.

        Throws TypeError if the configuration contains any non-serializable types.
        :param cfg: The experiment configuration represented by a dictionary.
        """
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
        """
        Writes the result of an experiment to the database.

        Throws TypeError if the configuration contains any non-serializable types.
        :param result: The experiment result represented by a dictionary.
        """
        with open(os.path.join(self.out_path, "results.jsonl"), "a", 1) as out_file:
            out_file.write(json.dumps(result, sort_keys=True, default=self.__json_encoder__) + "\n")


class MongoDBStorageBackend(StorageBackend):
    def __init__(self, mongo_host: str, mongo_port: int, mongo_database: str):
        """
        Initializes a storage backend which used MongoDB to store experiments and their results.

        :param mongo_host: The host of the MongoDB server.
        :param mongo_port: The port of the MongoDB server.
        :param mongo_database: The name of the database, which should be used to store results.
        """

        # Save properties.
        self.host = mongo_host
        self.port = mongo_port
        self.database = mongo_database
        self.client = None

    def __getstate__(self):
        """
        Constructs a state dict of this object. Mainly used to appropriately support pickling.
        :return: Dictionary holding the state of this particular storage backend instance.
        """
        return {"host": self.host, "port": self.port, "database": self.database, "client": None}

    def write_experiment_config(self, cfg: dict):
        """
        Writes the configuration of an experiment to the database.

        Throws TypeError if the configuration contains any non-serializable types.
        :param cfg: The experiment configuration represented by a dictionary.
        """
        if self.client is None:
            self.client = MongoClient(self.host, self.port)
        self.client[self.database]["experiments"].insert_one(json.loads(json.dumps(cfg, default=self.__json_encoder__)))  # dumping + loading ensures, that no JSON-incompatible objects are saved to MongoDB

    def add_result(self, result: dict):
        """
        Writes the result of an experiment to the database.

        Throws TypeError if the configuration contains any non-serializable types.
        :param result: The experiment result represented by a dictionary.
        """
        if self.client is None:
            self.client = MongoClient(self.host, self.port)
        self.client[self.database]["results"].insert_one(json.loads(json.dumps(result, default=self.__json_encoder__)))  # dumping + loading ensures, that no JSON-incompatible objects are saved to MongoDB
