import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import torch
from Prompter.utils import *
from Prompter.models import instructir
from Prompter.text.models import LanguageModel, LMHead

class InstructIRProcessor:
    def __init__(self, config_path="Prompter/configs/eval5d.yml", image_model_path="Prompter/models/im_instructir-7d.pt", lm_model_path="Prompter/models/lm_instructir-7d.pt", seed=42):
        self.device = torch.device("cpu")
        self.config_path = config_path
        self.image_model_path = image_model_path
        self.lm_model_path = lm_model_path
        self.seed = seed
        self.config = None
        self.model = None
        self.language_model = None
        self.lm_head = None
        self.initialize()

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)

    def load_model(self):
        print("Creating InstructIR")
        cfg = self.config.model
        self.model = instructir.create_model(input_channels=cfg.in_ch, width=cfg.width, enc_blks=cfg.enc_blks,
                                             middle_blk_num=cfg.middle_blk_num, dec_blks=cfg.dec_blks, txtdim=cfg.textdim)
        self.model = self.model.to(self.device)
        print("IMAGE MODEL CKPT:", self.image_model_path)
        self.model.load_state_dict(torch.load(self.image_model_path, map_location=self.device), strict=True)
        print("Loaded weights!")

    def load_language_model(self):
        if self.config.model.use_text:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            LMODEL = self.config.llm.model
            self.language_model = LanguageModel(model=LMODEL)
            self.lm_head = LMHead(embedding_dim=self.config.llm.model_dim, hidden_dim=self.config.llm.embd_dim, num_classes=self.config.llm.nclasses)
            print("LMHEAD MODEL CKPT:", self.lm_model_path)
            self.lm_head.load_state_dict(torch.load(self.lm_model_path, map_location=self.device), strict=True)
            print("Loaded weights!")
        else:
            self.language_model = None
            self.lm_head = None

    def initialize(self):
        self.seed_everything()
        self.load_config()
        self.load_model()
        self.load_language_model()

    def process_img(self, image, prompt):
        y = torch.Tensor(image).permute(2,0,1).unsqueeze(0)
        lm_embd = self.language_model(prompt)
        text_embd, deg_pred = self.lm_head(lm_embd)
        x_hat = self.model(y, text_embd)
        restored_img = x_hat[0].permute(1,2,0).cpu().detach().numpy()
        restored_img = np.clip(restored_img, 0., 1.)
        return restored_img

    @staticmethod
    def plot_all(images, names=["Before", "After"], figsize=(10, 5)):
        plt.figure(figsize=figsize)
        for i, (img, name) in enumerate(zip(images, names)):
            plt.subplot(1, len(images), i+1)
            plt.title(name)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

# Example usage
if __name__ == "__main__":
    processor = InstructIRProcessor()
    IMG = "Prompter/images/rain-020.png"
    PROMPT = "Retouch the picture as a professional photographer please"
    image = load_img(IMG)
    restored_image = processor.process_img(image, PROMPT)
    processor.plot_all([image, restored_image], names=["Before", "After"])
