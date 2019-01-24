# test the pre-trained model on a single video
# (working on it)
# Bolei Zhou and Alex Andonian

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torch.nn.parallel
import torch.optim
from models import TSN
from transforms import *
import datasets_video
from torch.nn import functional as F


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
#group = parser.add_mutually_exclusive_group(required=True)
#group.add_argument('--video_file', type=str, default=None)
#group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='moments',
                    choices=['something', 'jester', 'moments'])
#parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weight', type=str)

args = parser.parse_args()

# Get dataset categories.
categories_file = 'pretrain/{}_categories.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
net = TSN(num_class,
    args.test_segments,
    args.modality,
    base_model=args.arch,
    consensus_type=args.consensus_type,
    img_feature_dim=args.img_feature_dim, print_spec=False)


weights = args.weight
checkpoint = torch.load(weights)
#print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
net.cuda().eval()


# Load splits
with open('/data/datasets/EEG_VIDS/videos/list_test.txt', 'r') as f:
    filenames = f.readlines()
FRAME_ROOT = '/data/datasets/EEG_VIDS/video_frames'

# Initialize frame transforms.

transform = torchvision.transforms.Compose([
    GroupOverSample(net.input_size, net.scale_size),
    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    GroupNormalize(net.input_mean, net.input_std),
])




for vid_file in filenames:
    
    with torch.no_grad():

	frame_folder = os.path.join(FRAME_ROOT,vid_file.split('./')[-1].split()[0]) #there was an extra \n
	#print frame_folder

	# Obtain video frames
	#if args.frame_folder is not None:
	print('Loading frames in %s'%frame_folder)
	import glob
	# here make sure after sorting the frame paths have the correct temporal order
	frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.jpg')))
	#print(frame_paths)
	#print(os.path.join(frame_folder, '*.jpg'))
	frames = load_frames(frame_paths)
	#~ else:
	#~ print('Extracting frames using ffmpeg...')
	#~ frames = extract_frames(vid_file, args.test_segments)



	# Make video prediction.
	data = transform(frames)
	input_var = torch.autograd.Variable(data.view(-1, 3, data.size(1), data.size(2)),
					volatile=True).unsqueeze(0).cuda()
	logits = net.forward(input_var)
	h_x = torch.mean(F.softmax(logits, 1), dim=0).data
	probs, idx = h_x.sort(0, True)

	# Output the prediction.
	video_name = vid_file
	print('RESULT ON ' + video_name)
	for i in range(0, 5):
	    print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
	    
	    
