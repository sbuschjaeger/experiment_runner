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
import torch

from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from ptflops import get_model_complexity_info

# from Models import SKLearnModel
from Utils import store_model, flatten_dict, replace_objects, Scale,dict_to_str
from BaggingClassifier import BaggingClassifier
# from BinarisedNeuralNetworks import binarize, BinaryLinear, BinaryConv2d, BinaryTanh
from NNEnsemble import NNEnsemble
# from DiverseNN import DiverseNN
# from GradientBoostedNets import GradientBoostedNets

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

def store(model, dirpath, bcfg, mcfg, accuracy):
    ensemble = dict()
    #ensemble["name"] = modelname
    ensemble["accuracy"] = accuracy
    ensemble["model"] = []
    
    # TODO THIS IS ONLY A CONVENTION TO USE "BINARY" AND "FLOAT". I SHOULD REALLY LOOK AT THIS
    #template = partial(mcfg["base_model"],mcfg["base_type"].replace("binary", "float"))

    if not hasattr(model, "n_estimators"):
        store_model(model.layers_, dirpath + "/model.onnx",bcfg["dim"], bcfg["verbose"])
        
        if isinstance(model.layers_[-1], Scale):
            weight = model.layers_[-1].scale.item()
        else:
            weight = 1.0
        
        ensemble["model"].append(
            {
                "type":"NeuralNetwork", 
                "file": dirpath + "/model.onnx",
                "weight":weight
            }
        )
    else:
        for i, est in enumerate(model.estimators_):
            filename = "base{}.onnx".format(i)
            filepath = dirpath + "/" + filename

            if isinstance(model, AdaBoostClassifier):
                if isinstance(est.layers_[-1], Scale):
                    weight = est.layers_[-1].scale.item()*model.estimator_weights_[i]
                else:
                    weight = model.estimator_weights_[i]

                # base_type = base_type.replace("binary", "float")
                store_model(est.layers_, filepath,bcfg["dim"], bcfg["verbose"])
            elif isinstance(model, BaggingClassifier):
                if isinstance(est.layers_[-1], Scale):
                    weight = est.layers_[-1].scale.item()*1.0/model.n_estimators
                else:
                    weight = 1.0/model.n_estimators

                # base_type = base_type.replace("binary", "float")
                store_model(est.layers_, filepath,bcfg["dim"], bcfg["verbose"])
            else:
                if isinstance(est[-1], Scale):
                    weight = est[-1].scale.item()*1.0/model.n_estimators
                else:
                    weight = 1.0/model.n_estimators
                
                # base_type = base_type.replace("binary", "float")
                store_model(est, filepath,bcfg["dim"], bcfg["verbose"])
           
            ensemble["model"].append(
                {
                    "type":"NeuralNetwork", 
                    "file": filename,
                    "weight":weight
                }
            )
        
    with open(dirpath + "/model.json", 'w') as f:
        json.dump(ensemble, f)

def complexity(model, max_models = None):
    if not hasattr(model, "n_estimators"):
        norm = 0
        for m in model.layers_._modules.values():
            t_sum = 0
            if hasattr(m, 'weight'):
                t_sum += m.weight.norm(p="fro").item()
            
            if hasattr(m, 'bias'):
                t_sum += m.bias.norm(p="fro").item()
            
            if t_sum > 0:    
                norm += np.log(t_sum) #detach().cpu().numpy()
            # print("t_sum: ", t_sum )
            # print("norm: ", norm)
        return norm
    else:
        if max_models is None:
            max_models = model.n_estimators

        if isinstance(model, AdaBoostClassifier):
            weights = model.estimator_weights_[:max_models]
            weights = weights / np.sum(weights)
        else:
            weights = [1.0/max_models for _ in range(max_models)]

        norm = 0
        for i, est in enumerate(model.estimators_[:max_models]):
            w = weights[i]
            if isinstance(model, (AdaBoostClassifier, BaggingClassifier)):
                i_est = est.layers_
            else:
                i_est = est
            
            t_norm = 0
            for m in i_est._modules.values():
                t_sum = 0

                if hasattr(m, 'weight'):
                    t_sum += m.weight.norm(p="fro").item()
                
                if hasattr(m, 'bias'):
                    t_sum += m.bias.norm(p="fro").item()
                
                if t_sum > 0:        
                    t_norm += np.log(t_sum) 

            norm += w*t_norm
        return norm

def test_model(bcfg_original, experiment):
    experiment_id, mcfg = experiment
    lock.acquire()
    CUDA_DEVICE = cuda_devices_available.pop(0)
    lock.release()
    out_file = open(bcfg_original["out_path"] + "/results.jsonl","a",1)

    no_runs = bcfg_original.get("no_runs",1)
    accuracies = []
    fittimes = []
    for run_id in range(no_runs):
        if not "train_test_split" in bcfg_original:
            x_train,y_train = bcfg_original["data_loader"](bcfg_original["train"])
            x_test,y_test = bcfg_original["data_loader"](bcfg_original["test"])
        else:
            split_ratio = bcfg_original["train_test_split"]
            X,y = bcfg_original["data_loader"](bcfg_original["data"])
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42+run_id,stratify=y)

        bcfg = copy.deepcopy(bcfg_original)
        dirpath = bcfg["out_path"] + "/" + str(experiment_id) + "-" + str(run_id)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(dirpath + "/architecture.txt", 'w') as out:
            if "base_estimator" in mcfg:
                base_fun = mcfg["base_estimator"]
                if not inspect.isclass(base_fun):
                    if isinstance(base_fun, partial):
                        lines = inspect.getsource(base_fun.func)
                    else:
                        lines = inspect.getsource(base_fun)
                    out.write(lines)
            else:
                # TODO!
                pass 
        
        with open(dirpath + "/config.json", 'w') as out:
            out.write(json.dumps(replace_objects(mcfg)))

        if bcfg["verbose"]:
            tmp_dict = replace_objects(mcfg.copy())
            json_str = json.dumps(tmp_dict, indent=4)
            print("STARTING {}-{} ON CUDA_DEVICE {}. Config is {} \n".format(experiment_id, run_id, CUDA_DEVICE, json_str))

        tmp_cfg = copy.deepcopy(mcfg)
        tmp_cfg["out_path"]  = bcfg["out_path"] + "/" + str(experiment_id) + "-" + str(run_id) 
        tmp_cfg["verbose"]  = bcfg["verbose"]
        tmp_cfg["x_test"] = x_test
        tmp_cfg["y_test"] = y_test

        if "base_parameters" in mcfg:
            base_parameters = mcfg["base_parameters"]
            if mcfg["model"].__name__ == "AdaBoostClassifier":
                base_model = mcfg["base_estimator"](**base_parameters)
            else:
                base_model = partial(mcfg["base_estimator"], **base_parameters)
            tmp_cfg["base_estimator"] = base_model
        else:
            base_parameters = {}
        
        base_parameters["x_test"] = x_test
        base_parameters["y_test"] = y_test
        base_parameters["verbose"]  = bcfg["verbose"]
        base_parameters["out_path"]  = bcfg["out_path"] + "/" + str(experiment_id) + "-" + str(run_id)
        
        tmp_cfg.pop("model")
        model = mcfg["model"](**tmp_cfg)
        with torch.cuda.device(CUDA_DEVICE):
            print("CHECKING {}".format(base_parameters["out_path"] + "/model.checkpoint"))
            if os.path.isfile(base_parameters["out_path"] + "/model.checkpoint"):
                if bcfg["verbose"]:
                    print("MODEL FOUND. USING THAT")

                # TODO This should only be called if necessary / be part of an init method or something. I dont know.
                model.layers_ = model.base_estimator()
                model.classes_ = 0
                model.n_classes_ = 0
                model.y_ = None
                model.X_ = None
                model.load_state_dict(torch.load(base_parameters["out_path"] + "/model.checkpoint", map_location='cpu'))
                model.eval()
                model.cuda()
                # TODO: Retrain model here
                # print("LOADED")
                # print(model)
                fit_time = 0
            else:    
                start_time = time.time()
                model.fit(x_train, y_train)
                fit_time = time.time() - start_time

            # dummy_model = model.estimators_[0] if hasattr(model, "estimators_") else model
            # base_param, base_mac = get_model_complexity_info(
            #     dummy_model, 
            #     bcfg["dim"], 
            #     as_strings=False, 
            #     print_per_layer_stat=False
            # )

            if hasattr(model, "staged_predict_proba"):
                preds_train = [(i+1,pred) for i, pred in enumerate(model.staged_predict_proba(x_train))]
                preds_test = [(i+1,pred) for i, pred in enumerate(model.staged_predict_proba(x_test))]
            else:
                ne = mcfg["n_estimators"] if "n_estimators" in mcfg else 1
                preds_test = [(ne, model.predict_proba(x_test))]
                preds_train = [(ne, model.predict_proba(x_train))]

            accuracy = None
            for (ne, pred_test), (_, pred_train) in zip(preds_test, preds_train):
                accuracy_test = accuracy_score(np.argmax(pred_test, axis=1),y_test)*100.0
                accuracy_train = accuracy_score(np.argmax(pred_train, axis=1),y_train)*100.0
                
                tmp_dict = replace_objects(mcfg.copy())
                if "metrics" in bcfg:
                    for m_name,m_fun in bcfg["metrics"]:
                        #tmp_dict[m_name + "_train"] = m_fun(model, ne, x_train, y_train) 
                        tmp_dict[m_name + "_test"] = m_fun(model, ne, x_test, y_test) 

                # TODO REFACTOR METRICS
                tmp_dict["experiment_id"] = experiment_id
                tmp_dict["run_id"] = run_id
                tmp_dict["accuracy_train"] = accuracy_train
                tmp_dict["accuracy_test"] = accuracy_test
                # tmp_dict["complexity"] = complexity(model, ne)
                # tmp_dict["MAC"] = base_mac*ne + ne
                # tmp_dict["param"] = base_param*ne + ne
                tmp_dict["fit_time"] = fit_time

                lock.acquire()
                out_file.write(json.dumps(tmp_dict) + "\n")
                lock.release()
                accuracy = accuracy_test
                fit_time = fit_time
            accuracies.append(accuracy)
            fittimes.append(fit_time)

            if bcfg["store_model"]:
                if bcfg["verbose"]:
                    print("STORING MODEL IN  {}".format(dirpath))
                store(model, dirpath, bcfg, mcfg, accuracy)
                torch.save(model.state_dict(), dirpath + "/model.checkpoint")

        lock.acquire()
        cuda_devices_available.append(CUDA_DEVICE)
        lock.release()

    return experiment_id,np.mean(accuracies),np.mean(fittimes)

def run_experiments(basecfg, models, n_cores = 8):
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
            # for the_file in os.listdir(basecfg["out_path"]):
            #     file_path = os.path.join(basecfg["out_path"], the_file)
            #     if os.path.isfile(file_path):
            #         os.unlink(file_path)
            #     elif os.path.isdir(file_path): 
            #         shutil.rmtree(file_path)

        l = multiprocessing.Lock()
        
        manager = Manager()
        shared_list = manager.list(basecfg["cuda_devices"])
        print("Starting {} experiments on {} cores using {} GPUs".format(len(models), n_cores, len(set(basecfg["cuda_devices"]))))

        models = [(i, m) for i, m in enumerate(models)]
        pool = MyPool(n_cores, initializer=init, initargs=(l,shared_list))
        for run_id, acc,fit_time in pool.imap(partial(test_model, basecfg), models):
            results.append((run_id,acc, fit_time))
            print("{:02d}/{} ACC: {:.4f} TIME: {:.2f}s".format(run_id+1,len(models),acc,fit_time))

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
            # msg["From"] = "sebastian.buschjaeger@postamt.cs.uni-dortmund.de"
            msg["To"] = basecfg["mail"]
            if "name" in basecfg:
                msg["Subject"] = "{} finished on {}".format(basecfg["name"], socket.gethostname())
            else:
                results = sorted(results, key=lambda x: x[2])

                return_str += "\nrun_id\taccuracy\tfit_time\n"
                return_str += "\n".join(["{}\t{:6.4f}\t{:6.2f}".format(x[0],x[1],x[2]) for x in results]) 
                msg["Subject"] = "Experiments finished on {}".format(socket.gethostname())  

            msg.attach(MIMEText(return_str, "plain"))
            server.sendmail(msg['From'], msg['To'], msg.as_string())
        
