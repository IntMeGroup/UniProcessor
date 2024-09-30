import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--warmup_epochs', type=int, default=12, help='number of warmup epochs to train the model.')
parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="uniprocessor",help = "Determine to log to wandb or not and the project name")
# parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 4,help = "Number of GPUs to use for training")

# add
parser.add_argument('--arch', type=str, default ='uniprocessor_blip_tiny',  help='archtechture')
parser.add_argument('--output_dir', type=str, default="output/", help='output save path')
parser.add_argument("--ckpt_dir",type=str,default="output/train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument('--logger', type=str, default='csvlogger', help="logger format") # wblogger,tblogger,csvlogger
parser.add_argument('--log_name', type=str, default ='uniprocessor')
parser.add_argument('--resume', action='store_true',default=False)
parser.add_argument('--pretrain_weights',type=str, default='output/train_ckpt', help='path of pretrained_weights')
parser.add_argument('--dataset', type=str, default ='SIDD')
parser.add_argument('--mixup_aug', action='store_true',default=False)
parser.add_argument('--data_path', type=str, default ='./datasets/SIDD/train',  help='dir of train data')
parser.add_argument('--input_size', type=int, default=128, help='patch size of training sample')

options = parser.parse_args()