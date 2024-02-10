import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import time

class DiffusionModel:
    def __init__(self, image_name="result.png", config='config/underwater.json', phase='val', gpu_ids=None, debug=False, log_infer=False, mode=1):
        if mode != 1:
            print("using mode 2")
            self.config = 'config/underwater_video.json'
        else:
            self.config = config
        self.phase = phase
        self.gpu_ids = gpu_ids
        self.debug = debug
        self.log_infer = log_infer
        self.image_name = image_name
        self.mode = mode
        self.parse_configs()
        self.setup_logging()
        self.initialize_dataset()
        self.initialize_model()

    def parse_configs(self):
        # parse configs
        self.opt = Logger.parse(self)
        # Convert to NoneDict, which return None for missing key.
        self.opt = Logger.dict_to_nonedict(self.opt)

    def setup_logging(self):
        # logging
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        Logger.setup_logger(None, self.opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', self.opt['path']['log'], 'val', level=logging.INFO)
        self.logger = logging.getLogger('base')
        self.logger.info(Logger.dict2str(self.opt))

    def initialize_dataset(self):
        # dataset
        for phase, dataset_opt in self.opt['datasets'].items():
            if phase == 'val':
                self.val_set = Data.create_dataset(dataset_opt, phase)
                self.val_loader = Data.create_dataloader(
                    self.val_set, dataset_opt, phase)
        self.logger.info('Initial Dataset Finished')

    def initialize_model(self):
        # model
        self.diffusion = Model.create_model(self.opt)
        self.logger.info('Initial Model Finished')

        self.diffusion.set_new_noise_schedule(
            self.opt['model']['beta_schedule']['val'], schedule_phase='val')

    def run(self):
        self.logger.info('Begin Model Inference.')
        current_step = 0
        current_epoch = 0
        idx = 0

        result_path = '{}'.format(self.opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(self.val_loader):
            idx += 1
            self.diffusion.feed_data(val_data)
            start = time.time()
            self.diffusion.test(continous=True)
            end = time.time()
            print('Execution time:', (end - start), 'seconds')
            visuals = self.diffusion.get_current_visuals(need_LR=False)

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    #Metrics.save_img(Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
                    Metrics.save_img(Metrics.tensor2img(sr_img[iter]), 'result/result.png')

            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                if self.mode == 1:
                    Metrics.save_img(Metrics.tensor2img(visuals['SR'][-1]), f'./temp/enhance/{self.image_name}')
                else:
                    video_image_name = "{:05d}.png".format(idx)
                    Metrics.save_img(Metrics.tensor2img(visuals['SR'][-1]), f'./temp/enhance_video/{video_image_name}')
                # for i in range(len(visuals['SR'])):
                #     Metrics.save_img(
                #         Metrics.tensor2img(visuals['SR'][i]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, str(i)))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

if __name__ == "__main__":
    model = DiffusionModel()
    model.run()
