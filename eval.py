import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, load_model_test, load_gan_discrimitor, load_gan_generator_test
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils.metrics import Evaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


test_loader = get_test_loaders(opt,batch_size=1)

model = load_model_test(opt, dev)
path_cd = 'tmp/checkpoint_cd_epoch_best.pt'   # the path of the model
model.load_state_dict(torch.load(path_cd, map_location='cpu'))

G_AB = load_gan_generator_test(opt, dev)
path_g_ab = 'tmp/checkpoint_gab_epoch_best.pt'
G_AB.load_state_dict(torch.load(path_g_ab, map_location='cpu'))

G_BA = load_gan_generator_test(opt, dev)
path_g_ba = 'tmp/checkpoint_gba_epoch_best.pt'
G_BA.load_state_dict(torch.load(path_g_ba, map_location='cpu'))

model.eval()
G_AB.eval()
G_BA.eval()

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

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

evaluator = Evaluator(opt.num_class)
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        real_A = Variable(batch_img1.type(Tensor)).cuda()
        real_B = Variable(batch_img2.type(Tensor)).cuda()

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
        
        [cd_preds_1, cd_preds_2, cd_preds_3, cd_preds, cd_preds_fusion] = model(real_A_norm2, real_B_norm2, fake_B_norm2, fake_A_norm2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds_1 = cd_preds_1[-1]
        _, cd_preds_1 = torch.max(cd_preds_1, 1)
        cd_preds_2 = cd_preds_2[-1]
        _, cd_preds_2 = torch.max(cd_preds_2, 1)
        cd_preds_3 = cd_preds_3[-1]
        _, cd_preds_3 = torch.max(cd_preds_3, 1)
        cd_preds_fusion = cd_preds_fusion[-1]
        _, cd_preds_fusion = torch.max(cd_preds_fusion, 1)

        evaluator.add_batch(labels, cd_preds, cd_preds_1, cd_preds_2, cd_preds_3, cd_preds_fusion)

mIoU, mIoU_1, mIoU_2, mIoU_3, mIoU_fusion = evaluator.Mean_Intersection_over_Union()
P, P_1, P_2, P_3, P_fusion = evaluator.Precision()
R, R_1, R_2, R_3, R_fusion = evaluator.Recall()
F1, F1_1, F1_2, F1_3, F1_fusion = evaluator.F1()

print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P, R, F1, mIoU))
print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P_1, R_1, F1_1, mIoU_1))
print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P_2, R_2, F1_2, mIoU_2))
print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P_3, R_3, F1_3, mIoU_3))
print('Precision: {}\nRecall: {}\nF1-Score: {}\nmIoU: {}'.format(P_fusion, R_fusion, F1_fusion, mIoU_fusion))
