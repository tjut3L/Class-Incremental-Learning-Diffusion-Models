import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import json
import random
import sys
sys.path.append('/home/tjut_zhanghaiyang/paper-code/aa/sun/PyCIL-master/latant_diffusion_model')
from LDM.util import instantiate_from_config
from LDM.models.diffusion.ddim import DDIMSampler
from LDM.models.diffusion.plms import PLMSSampler



from utils.data_manager import DataManager

torch.cuda.set_device(4)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class Text2Img(object):
    def __init__(self, labels,
                #  coarse_labels,
                 class_order,
                 attribute,
                 outdir="data/cifar10", 
                 steps=200, 
                 use_plms=True, 
                 ddim_eta=1.0, 
                 n_sum=500, 
                 n_samples=16, 
                 scale=5.0, 
                 H=256, 
                 W=256):
        self.outdir = outdir
        self.steps = steps
        self.use_plms = use_plms
        self.ddim_eta = ddim_eta
        self.scale = scale
        self.H = H
        self.W = W
        self.iter=[n_samples]
        while n_sum-sum(self.iter) > n_samples:
            self.iter.append(n_samples)
        offset = n_sum - sum(self.iter)
        if offset > 0:
            self.iter.append(offset)
        self.config = OmegaConf.load("latant_diffusion_model/configs/latent-diffusion/txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = load_model_from_config(self.config, "latant_diffusion_model/models/ldm/text2img-large/model.ckpt")  # TODO: check path
        self.model = self.model.to(self.device)
        self.prompts = self.creat_prompt(labels,class_order,attribute)
        # self.prompts = self.creat_prompt(labels,coarse_labels)

    # def creat_prompt(self,labels,coarse_labels):
    def creat_prompt(self,labels,class_order, attribute):
        prompts={}
        for i in range(len(labels)):
            # prompt=[]
            # class_attribute = attribute[str(class_order[i])]
            # class_label = labels[i]
            # for attr in class_attribute:
            #     prompt.append("a photo of "+ labels[i] + " that " + attr)

            # prompt = "a photo of "+labels[i]+" that belong to "+coarse_labels[i]
            # prompt = labels[i]+", one type of "+coarse_labels[i]
            prompt = "a photo of " + labels[i]
            # prompt = "a photo of " + labels[i] + ", a type of "+coarse_labels[i]
            prompts[labels[i]] = prompt

        return prompts
    
    def generate_img(self):
        if self.use_plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)
        os.makedirs(self.outdir, exist_ok=True)
        for prompt in self.prompts.keys():
            # print(self.prompts[prompt])
            sample_path = os.path.join(self.outdir, prompt)
            os.makedirs(sample_path, exist_ok=True)
            base_count = len(os.listdir(sample_path))
            all_samples=list()
            with torch.no_grad():
                with self.model.ema_scope():
                    for i in trange(len(self.iter), desc="Sampling"):
                        choice_prompt = random.choice(self.prompts[prompt])
                        print(choice_prompt)
                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.iter[i] * [""])
                        # c = self.model.get_learned_conditioning(self.iter[i] * [self.prompts[prompt]])   #将文本转为特征[77,1280]
                        c = self.model.get_learned_conditioning(self.iter[i] * [choice_prompt])
                        shape = [4, self.H//8, self.W//8]
                        samples_ddim, _ = sampler.sample(S=self.steps,
                                                        conditioning=c,
                                                        batch_size=self.iter[i],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=self.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=self.ddim_eta)

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                            base_count += 1
                        all_samples.append(x_samples_ddim)
        print(f"Your samples are ready and waiting four you here: \n{self.outdir} \nEnjoy.")


if __name__ == "__main__":

    # json_file_path = 'data/class_label.json'
    # json_file_path = 'data/LABO_CIFAR100.json'

    # with open(json_file_path, 'r', encoding='utf-8') as file:
    #     json_data = json.load(file)

    # data_manager = DataManager("cifar100", True , 1993, 100, 0)
    # class_labels = data_manager._train_class_labels
    # class_order = data_manager._class_order

    # coarse_labels = data_manager._coarse_labels
    data_manager = DataManager("cifar10", True , 1993, 10, 0)
    class_labels = data_manager._train_class_labels
    class_order = data_manager._class_order

    # t2i=Text2Img(labels=class_labels,
    #         coarse_labels=coarse_labels,
    #         outdir="data/plms/cifar100-photo", 
    #         steps=50, 
    #         use_plms=True, 
    #         ddim_eta=0.0, 
    #         n_sum=500, 
    #         n_samples=40, 
    #         scale=5.0, 
    #         H=256, 
    #         W=256
    #         )

    # t2i=Text2Img(labels=class_labels,
    #         class_order=class_order,
    #         attribute=json_data,
    #         outdir="data/plms/cifar100-photo-attribute", 
    #         steps=50, 
    #         use_plms=True, 
    #         ddim_eta=0.0, 
    #         n_sum=500, 
    #         n_samples=40, 
    #         scale=5.0, 
    #         H=256, 
    #         W=256
    #         )
    t2i=Text2Img(
            labels=class_labels,
            class_order=class_order,
            attribute=[],
            outdir="data/cifar10", 
            steps=50, 
            use_plms=True, 
            ddim_eta=0.0, 
            n_sum=500, 
            n_samples=40, 
            scale=5.0, 
            H=256, 
            W=256
            )
    t2i.generate_img()
