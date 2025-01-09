import sys
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    history_cnn = []
    history_nme = []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            cnn_task_acc=[]
            cnn_task_forgets=[]
            nme_task_acc=[]
            nme_task_forgets=[]
            for i in range(task):
                cnn_ta = cnn_accy["grouped"][str(i)]
                nme_ta = nme_accy["grouped"][str(i)]
                cnn_tf = max(np.array(history_cnn)[:,i]) - cnn_ta
                nme_tf = max(np.array(history_nme)[:,i]) - nme_ta
                cnn_task_acc.append(cnn_ta)
                nme_task_acc.append(nme_ta)
                cnn_task_forgets.append(cnn_tf)
                nme_task_forgets.append(nme_tf)
            cnn_task_acc.append(cnn_accy["grouped"][str(task)])
            nme_task_acc.append(nme_accy["grouped"][str(task)])

            cnn_task_acc.extend([0]*(data_manager.nb_tasks-1-task))
            nme_task_acc.extend([0]*(data_manager.nb_tasks-1-task))

            logging.info("Forgettings (CNN): {}".format(cnn_task_forgets))
            if len(cnn_task_forgets) != 0:
                logging.info("Average Forgetting (CNN): {}".format(sum(cnn_task_forgets)/len(cnn_task_forgets)))
            if len(nme_task_forgets) != 0:
                logging.info("Average Forgetting (NME): {}".format(sum(nme_task_forgets)/len(nme_task_forgets)))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))

            history_cnn.append(cnn_task_acc)
            history_nme.append(nme_task_acc)
            

        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            task_acc=[]
            task_forgets=[]
            for i in range(task):
                ta = cnn_accy["grouped"][str(i)]
                tf = max(np.array(history_cnn)[:,i]) - ta
                task_acc.append(ta)
                task_forgets.append(tf)
            task_acc.append(cnn_accy["grouped"][str(task)])
            task_acc.extend([0]*(data_manager.nb_tasks-1-task))

            logging.info("Forgettings (CNN): {}".format(task_forgets))
            if len(task_forgets) != 0:
                logging.info("Average Forgetting (CNN): {}".format(sum(task_forgets)/len(task_forgets)))

            
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        
            history_cnn.append(task_acc)
        
            

        


    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
