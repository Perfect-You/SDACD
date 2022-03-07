'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics, load_model_test,load_gan_generator_test
import os
from tqdm import tqdm
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
opt.visual = True

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)

# test_loader = get_test_loaders(opt)

model = load_model_test(opt, dev)
path_cd = 'tmp-ia/checkpoint_cd_epoch_20.pt'   # the path of the model
model.load_state_dict(torch.load(path_cd, map_location='cpu'))

G_AB = load_gan_generator_test(opt, dev)
path_g_ab = 'tmp-ia/checkpoint_gab_epoch_20.pt'
G_AB.load_state_dict(torch.load(path_g_ab, map_location='cpu'))

G_BA = load_gan_generator_test(opt, dev)
path_g_ba = 'tmp-ia/checkpoint_gba_epoch_20.pt'
G_BA.load_state_dict(torch.load(path_g_ba, map_location='cpu'))

def unnormalize(tensor):
  tensor = tensor.clone()  # avoid modifying tensor in-place

  def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

  def norm_range(t):
    norm_ip(t, float(t.min()), float(t.max()))

  norm_range(tensor)

  return tensor

transform1 = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])


model.eval()
G_AB.eval()
G_BA.eval()

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, name in tbar:
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        real_A = Variable(batch_img1.type(Tensor)).cuda()
        real_B = Variable(batch_img2.type(Tensor)).cuda()

        # Get predictions and calculate loss
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        real_A_norm2 = unnormalize(real_A)
        real_A_norm2 = transform1(real_A_norm2)
        real_B_norm2 = unnormalize(real_B)
        real_B_norm2 = transform1(real_B_norm2)
        fake_A_norm2 = unnormalize(fake_A)
        fake_A_norm2 = transform1(fake_A_norm2)
        fake_B_norm2 = unnormalize(fake_B)
        fake_B_norm2 = transform1(fake_B_norm2)

        [cd_preds_1, cd_preds_2, cd_preds_3, cd_preds] = model(real_A_norm2, real_B_norm2, fake_B_norm2, fake_A_norm2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        file_path = './output_img/' + str(name[0])
        cv2.imwrite(file_path + '.png', cd_preds)

        index_img += 1
