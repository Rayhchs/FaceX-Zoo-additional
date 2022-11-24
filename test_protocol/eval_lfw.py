import os
import sys
import argparse
import yaml
from torch.utils.data import DataLoader
sys.path.append('../../test_protocol/')
from lfw.pairs_parser import PairsParserFactory
from lfw.lfw_evaluator import LFWEvaluator
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('../../')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory


def accu_key(elem):
    return elem[1]

def evaluation(test_set: str, data_conf_file: str, backbone_type: str, backbone_conf_file: str, model_path: str, batch_size=512, head_type=None):

    # parse config.
    with open(data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)[test_set]
        pairs_file_path = data_conf['pairs_file_path']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']
    # define pairs_parser_factory
    pairs_parser_factory = PairsParserFactory(pairs_file_path, test_set)

    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False), 
                             batch_size=batch_size, num_workers=4, shuffle=False)
    #model def
    backbone_factory = BackboneFactory(backbone_type, backbone_conf_file, head_type)
    model_loader = ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor('cuda:0')
    lfw_evaluator = LFWEvaluator(data_loader, pairs_parser_factory, feature_extractor, cropped_face_folder)
    if os.path.isdir(model_path):
        accu_list = []
        model_name_list = os.listdir(model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                model_path = os.path.join(model_path, model_name)
                model = model_loader.load_model(model_path)
                mean, std = lfw_evaluator.test(model)
                accu_list.append((os.path.basename(model_path), mean, std))
        accu_list.sort(key = accu_key, reverse=True)
    else:
        model = model_loader.load_model(model_path)
        mean, std = lfw_evaluator.test(model)
        accu_list = [(os.path.basename(model_path), mean, std)]

    return mean
    # pretty_tabel = PrettyTable(["model_name", "mean accuracy", "standard error"])
    # for accu_item in accu_list:
    #     pretty_tabel.add_row(accu_item)
    # print(pretty_tabel)

# if __name__ == '__main__':
#     evaluation(test_set='LFW', data_conf_file='data_conf.yaml', backbone_type='TF-NAS', backbone_conf_file='backbone_conf.yaml',
#         model_path=r'C:\Users\user\Desktop\FaceX-Zoo\training_mode\conventional_training\out_dir\Epoch_1.pt', batch_size=16)