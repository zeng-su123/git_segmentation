import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='PAIP 2020 Challenge - CRC Prediction', formatter_class=SmartFormatter)

# parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--gpu", type=str, default="0")  # gpu的设置，仅有一块gpu~1080Ti,因此默认为0
parser.add_argument("--seed", type=int, default=2020)  # 随机数种子
parser.add_argument('--output_dir', type=str, default=r"H:\output_dir",
                    help='Where progress/checkpoints will be saved')  # 训练模型结果保存的位置

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')  # 设置的epoch ==150 数量
# parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size for training')  # batchsize 的设置，通过实验发现，仅仅能使用16
parser.add_argument('--data_fold', type=str, help='Which training Fold (Cross Validation)')
# parser.add_argument('--num_classes', type=int, default=1, help='Model output neurons')
parser.add_argument('--num_classes', type=int, default=4, help='Model output neurons')  # 一共有多少个类别， 4类， 背景，左心室，右心室，左心室心肌
parser.add_argument('--data_fold_validation', type=str,
                    help='Which testing Fold (Only when folding by vendor)')  # 确定哪个测试文件夹 ，什么意思？
parser.add_argument('--fold_system', type=str,
                    help='How to create data folds')  # 什么叫做怎么创建数据文件夹  可选 ”vendor" , “patient”，“all”
parser.add_argument('--dataset', type=str, help='Dataset to use')  # 指定使用的数据集
parser.add_argument('--label_type', type=str,
                    help='"mask" for segmentation or "vendor_label" for classification')  # 标签类别，分割和分类两种标签

parser.add_argument('--model_name', type=str, default='resnet34_unet_scratch',
                    help='Model name for training')  # 模型的名字，根据模型的名字来区分训练的模型是什么。
# parser.add_argument('--model_name', type=str, default='resnet34', help='Model name for training')
# parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')  # 指定数据集增强的方式
parser.add_argument('--data_augmentation', type=str, default="combination_old",
                    help='Apply data augmentations at train time')  # 指定数据集增强的方式
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')  # 数据集裁剪的大小
parser.add_argument('--img_size', type=int, default=224, help='Final img squared size')  # 最后的图像大小

parser.add_argument('--binary_threshold', type=float, default=0.5, help='Threshold for masks probabilities')

parser.add_argument('--criterion', type=str, default='bce_dice_border_ce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='default',
                    help='Weights for each subcriterion')  # weights_criterion 加权指标 ，每个子指标的权重 ，比如coress-entropy_loss soft-cross-entropy-loss

parser.add_argument('--model_checkpoint', type=str, default="",
                    help='Where is the model checkpoint saved')  # 模型保存的位置，和output_dir有什么区别呢？ anwser 区别就是，用来断点续训
parser.add_argument('--segmentator_checkpoint', type=str, default="",
                    help='Segmentator checkpoint (predict v2)')  # 分割的checkpoint是什么？
parser.add_argument('--discriminator_checkpoint', type=str, default="",
                    help='Dicriminator checkpoint (predict v2)')  # 鉴别器的checkpoint保存位置，鉴别器是用来干嘛的？
parser.add_argument('--defrost_epoch', type=int, default=-1,
                    help='Number of epochs to defrost the model')  # 解冻模型的epoch数，即第几个epoch会解冻模型。

# parser.add_argument('--normalization', type=str, required=True, help='Data normalization method')
parser.add_argument('--normalization', type=str, default="reescale", required=True,
                    help='Data normalization method')  # normalization 的方式

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')  # 优化器的方法，使用adam
parser.add_argument('--scheduler', type=str, default="", help='Where is the model checkpoint saved')  # 学习率衰减策略
# parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')  # 设置学习率
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')  # 设置学习率
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimun Learning rate')  # 最低学习率
parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum Learning rate')  # 最高学习率
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, help='Steps when steps scheduler choosed')

parser.add_argument('--add_depth', action='store_true',
                    help='If apply image transformation 1 to 3 channels or not')  # 与图像的通道数有关
parser.add_argument('--weakly_labelling', action='store_true',
                    help='Use weakly labels for class C')  # weakly_label ? 具体有什么用，不明白？ 与类别C有关

parser.add_argument('--apply_swa', action='store_true', help='Apply stochastic weight averaging')  # 优化策略有关.没有使用优化策略
parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')  # 优化策略有关
parser.add_argument('--swa_start', type=int, default=60, help='SWA_LR')  # 优化策略有关
parser.add_argument('--swa_lr', type=float, default=0.0001, help='SWA_LR')  # 优化策略有关

# parser.add_argument('--eval_overlays', action='store_true', help='Generate predictions overlays')
parser.add_argument('--eval_overlays', action='store_true',
                    help='Generate predictions overlays')  # 评估时的预测图片保存位置
parser.add_argument(
    '--eval_overlays_path', type=str, default='H:\eval_overlays',
    help='Where to save predictions overlays. If "none" no overlays are generated'
)

# For fold_eval.py  用来验证
# parser.add_argument('--evaluation_folder', type=str, default="",
#                     help='Folder to save evaluation results. If empty same as model path')  # 保存评估的结果
parser.add_argument('--evaluation_folder', type=str, default=r"K:\fold_eval",
                    help='Folder to save evaluation results. If empty same as model path')  # 保存评估的结果

parser.add_argument('--evaluation_descriptor', type=str, default="eval",
                    help='Subfolder name to save evaluation results')

# For prediction/submission  用来预测
parser.add_argument('--input_data_directory', type=str, default="H:\other_files\Heart_OpenDataset\OpenDataset\Testing",
                    help='Folder with volumes to predict')
parser.add_argument('--output_data_directory', type=str, default="H:\output_predict",
                    help='Folder to save prediction')  # 用来保存预测的结果

# For prediction v2 entropy adaptation   用来预测 v2
parser.add_argument('--target', type=str, default='B', help='Desired domain to transform')
parser.add_argument('--out_threshold', type=float, default=0.01, help='Difference stop condition')
parser.add_argument('--max_iters', type=int, default=100, help='Maximum number of iters to apply entropy')

parser.add_argument('--entropy_lambda', type=float, default=0.99, help='Learning rate')

parser.add_argument('--add_l1', action='store_true', help='If add L1 loss or not')
parser.add_argument('--l1_lambda', type=float, default=0.0, help='L1 impact factor')

parser.add_argument('--add_blur_param', action='store_true', help='Add blur matrix param or not')
parser.add_argument('--blur_lambda', type=float, default=0.0, help='Blur param impact factor')

parser.add_argument('--add_unblur_param', action='store_true', help='Add unblur matrix param or not')
parser.add_argument('--unblur_lambda', type=float, default=0.0, help='Unblur param impact factor')

parser.add_argument('--add_gamma_param', action='store_true', help='Add gamma param or not')
parser.add_argument('--gamma_lambda', type=float, default=0.0, help='Gamma param impact factor')

parser.add_argument('--segmentator_model_name', type=str, default='simple_unet', help='Segmentator model name')
parser.add_argument('--discriminator_model_name', type=str, default='simple_unet', help='Discriminator model name')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if args.output_data_directory == "":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # https://stackoverflow.com/a/55114771
    with open(args.output_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
