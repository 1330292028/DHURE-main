import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from net.a_net import a_net
from net.net import net
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
import numpy as np

parser = argparse.ArgumentParser(description='DHURE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='your dataset path')
parser.add_argument('--IHE_model', default='weights/IHE_weight.pth', help='Pretrained IHE model')
parser.add_argument('--PRD_model', default='weights/PRD_weight.pth', help='Pretrained PRD model')
parser.add_argument('--output_folder', type=str, default='test_results/')

opt = parser.parse_args()

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building models')
PRD_model = net().cuda()
PRD_model.load_state_dict(torch.load(opt.PRD_model, map_location=lambda storage, loc: storage))
IHE_model = a_net().cuda()
IHE_model.load_state_dict(torch.load(opt.IHE_model, map_location=lambda storage, loc: storage))
print('Pre-trained models is loaded.')

def eval():
    torch.set_grad_enabled(False)
    IHE_model.eval()
    PRD_model.eval()
    print('\nEvaluation:')
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()
        print(name)
        with torch.no_grad():
            L, R, X = PRD_model(input)
            EL, a = IHE_model(L)

            EI = EL * R
            I = EI
        if not os.path.exists(opt.output_folder):
            os.makedirs(opt.output_folder)
            os.mkdir(opt.output_folder + 'EI/')

        I = I.cpu()
        I_img = transforms.ToPILImage()(I.squeeze(0))
        I_img.save(opt.output_folder + '/EI/' + name[0])
    torch.set_grad_enabled(True)


eval()


