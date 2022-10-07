from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

config_file = 'custom_configs/hrnet_w_focal_config.py'
checkpoint_file = 'checkpoints/best_mIoU_iter_9600_w_palette.pth'

# config_file = 'configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
# checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img = 'bkk-urbanscapes-complete/val/image_0_0193.jpg'
result = inference_segmentor(model, img)

show_result_pyplot(model, img, result, get_palette('bangkokscapes'))


#----------------------insert palette into model checkpoint--------------------------------------

# import torch

# model = torch.load('checkpoints/best_mIoU_iter_9600.pth')
# model['meta']['PALETTE'] = [[128, 128, 0],[0, 0, 0],[128, 128, 128],[64, 0, 0],
#             [128, 0, 0],[0, 0, 128],[0, 128, 0],[0, 128, 128],
#             [64, 128, 0],[192, 0, 0],[128, 0, 128]]
# torch.save(model, 'checkpoints/best_mIoU_iter_9600_w_palette.pth')