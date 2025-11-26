import torch
from genie import VideoTokenizer

# Pre-assembled description of MagViT2
# encoder & decoder architecture
from genie import MAGVIT2_ENC_DESC
from genie import MAGVIT2_DEC_DESC

tokenizer = VideoTokenizer(
  # We can pass an arbitrary description of the
  # encoder architecture, see genie.tokenizer.get_module
  # to see which module are supported
  enc_desc = (
    ('causal-conv3d', { # A CausalConv3d layer
        'in_channels': 3,
        'out_channels': 64,
        'kernel_size': 3,
    }),
    ('video-residual', { # Residual Block
        'in_channels': 64,
        'kernel_size': 3,
        'downsample': (1, 2), # Optional down-scaling (time, space)
        'use_causal': True, # Using causal padding
        'use_blur': True, # Using blur-pooling
    }),
    ('video-residual', {
        'in_channels': 64,
        'out_channels': 128, # Output channels can be different
    }),
    ('video-residual', {
        'n_rep': 2, # We can repeat this block N-times
        'in_channels': 128,
    }),
    ('video-residual', {
        'in_channels': 128,
        'out_channels': 256, # We can mix different output channels...
        'kernel_size': 3,
        'downsample': 2, # ...with down-sampling (here time=space=2)
        'use_causal': True,
    }),
    ('causal-conv3d', { # Output project to quantization module
        'in_channels': 256,
        'out_channels': 18,
        'kernel_size': 3,
    }),
  ),
  # Save time, use a pre-made configuration!
  dec_desc = MAGVIT2_DEC_DESC,

  # Description of GAN discriminator
  disc_kwargs=dict(
      # Discriminator parameters
      inp_size = (64, 64), # Size of input frames
      model_dim = 64,
      dim_mults = (1, 2, 4),    # Channel multipliers
      down_step = (None, 2, 2), # Down-sampling steps
      inp_channels = 3,
      kernel_size = 3,
      num_groups = 8,
      act_fn = 'leaky', # Use LeakyReLU as activation function
      use_blur = True,  # Use BlurPooling for down-sampling
      use_attn = True,  # Discriminator can have spatial attention
      num_heads = 4,    # Number of (spatial) attention heads
      dim_head = 32,    # Dimension of each spatial attention heads
  ),

  # Keyword for the LFQ module
  d_codebook = 18, # Codebook dimension, should match encoder output channels
  n_codebook = 1, # Support for multiple codebooks
  lfq_bias = True,
  lfq_frac_sample = 1.,
  lfq_commit_weight = 0.25,
  lfq_entropy_weight = 0.1,
  lfq_diversity_weight = 1.,
  # Keyword for the different loss
  perceptual_model = 'vgg16', # We pick VGG-16 for perceptual loss
  # Which layer should we record perceptual features from
  perc_feat_layers = ('features.6', 'features.13', 'features.18', 'features.25'),
  gan_discriminate='frames', # GAN discriminator looks at individual frames
  gan_frames_per_batch = 4,  # How many frames to extract from each video to use for GAN
  gan_loss_weight = 1.,
  perc_loss_weight = 1.,
  quant_loss_weight = 1.,
)

batch_size = 4
num_channels = 3
num_frames = 16
img_h, img_w = 64, 64

# Example video tensor
mock_video = torch.randn(
  batch_size,
  num_channels,
  num_frames,
  img_h,
  img_w
)

# Tokenize input video
tokens, idxs = tokenizer.tokenize(mock_video)

# Tokenized video has shape:
# (batch_size, d_codebook, num_frames // down_time, H // down_space, W // down_space)

# To decode the video from tokens use:
rec_video = tokenizer.decode(tokens)

# To train the tokenizer (do many! times)
loss, aux_losses = tokenizer(mock_video)
loss.backward()