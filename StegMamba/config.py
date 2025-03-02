# Super parameters
clamp = 2.0
channels_in = 3
lr = 0.0002
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01
device_ids = [0]

# Train:
batch_size = 2
cropsize = 256
betas = (0.9, 0.999)
weight_step = 300
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 2
shuffle_val = False
val_freq = 1

# Dataset
TRAIN_PATH = '/train/'
VAL_PATH = '/valid/'
TEST_PATH = '/test/'


format_train = 'png'
format_val = 'png'
format_test = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False

# Saving checkpoints:
MODEL_PATH = ''
checkpoint_on_error = True
SAVE_freq = 1

IMAGE_PATH = ''
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_stego = IMAGE_PATH + 'stego/'
IMAGE_PATH_rsecret = IMAGE_PATH + 'rsecret/'
IMAGE_PATH_acover = IMAGE_PATH + 'acover/'
IMAGE_PATH_rcover = IMAGE_PATH + 'rcover/'
IMAGE_PATH_r = IMAGE_PATH + 'r/'
IMAGE_PATH_z = IMAGE_PATH + 'z/'

# Load:
suffix = ''
train_next = False
trained_epoch = 0
stage = ''
