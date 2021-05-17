import json
import os
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def write_experiment_config(self, cfg: dict):
        pass

    @abstractmethod
    def add_result(self, result: dict):
        pass


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

        # Check, whether the path exists otherwise create the directories..
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Write the config to disk.
        with open(os.path.join(out_path, "config.json"), 'w') as out:
            out.write(json.dumps(cfg, indent=4))

    def add_result(self, result: dict):
        with open(os.path.join(self.out_path, "results.jsonl"), "a", 1) as out_file:
            out_file.write(json.dumps(result, sort_keys=True) + "\n")