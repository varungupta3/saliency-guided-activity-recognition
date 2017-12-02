import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as trained_models
from collections import OrderedDict

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    # The network architecture for the encoder using VGG16 model weights pretrained on ImageNet data from the pytorch model zoo.
    vgg16 = trained_models.vgg16(pretrained=True)
    # Settting the Pretrained weights of the convolutional layers. vgg16.features.state_dict().keys() will give the 26 parameters (weights and biases of the conv)
    model = vgg16.features 
    # Extracting the First 3 convolutional layer and their parameters (weights and bias) to build our encoder model.
    encoder_fixed = nn.Sequential(*list(model.children())[:-14])
    # Fixing the paramters (weights and biases) of the first three layers
    for param in encoder_fixed.parameters():
      param.requires_grad = False
    # Generating the rest of the encoder model by extracting the pretrained paramters(training allowed).
    encoder_trainable = nn.Sequential(*list(model.children())[-14:-1])
    # Combining the two encoder setups to generate the complete architecture with trainable and not trainable parameters.
    total_layers = list(encoder_fixed.children())
    total_layers.extend(list(encoder_trainable.children()))

    # The complete encoder architecture
    self.encoder = nn.Sequential(*total_layers)

    # The network architecture for the decoder
    self.decoder = nn.Sequential(OrderedDict([

              ('uconv5_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_3', nn.ReLU(inplace=True)),              
              ('uconv5_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_2', nn.ReLU(inplace=True)),
              ('uconv5_1', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu5_1', nn.ReLU(inplace=True)),
              
              ('upool4', nn.modules.upsampling.Upsample(scale_factor=2,mode='nearest')),

              ('uconv4_3', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_3', nn.ReLU(inplace=True)),              
              ('uconv4_2', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_2', nn.ReLU(inplace=True)),
              ('uconv4_1', nn.Conv2d(512, 512, kernel_size=3,stride = 1, padding = 1)),
              ('relu4_1', nn.ReLU(inplace=True)),
              
              ('upool3', nn.modules.upsampling.Upsample(scale_factor=2,mode='nearest')),

              ('uconv3_3', nn.Conv2d(512, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_3', nn.ReLU(inplace=True)),              
              ('uconv3_2', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_2', nn.ReLU(inplace=True)),
              ('uconv3_1', nn.Conv2d(256, 256, kernel_size=3,stride = 1, padding = 1)),
              ('relu3_1', nn.ReLU(inplace=True)),
              
              ('upool2', nn.modules.upsampling.Upsample(scale_factor=2,mode='nearest')), 

              ('uconv2_2', nn.Conv2d(256, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_2', nn.ReLU(inplace=True)),
              ('uconv2_1', nn.Conv2d(128, 128, kernel_size=3,stride = 1, padding = 1)),
              ('relu2_1', nn.ReLU(inplace=True)),

              ('upool1', nn.modules.upsampling.Upsample(scale_factor=2,mode='nearest')),

              ('uconv1_2', nn.Conv2d(128, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1_2', nn.ReLU(inplace=True)),
              ('uconv1_1', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
              ('relu1_1', nn.ReLU(inplace=True)),

              ('output', nn.Conv2d(64, 1, kernel_size=1,stride = 1, padding = 0)),
              ('sigmoid',  nn.Sigmoid()) 

            ]))
  def forward(self,x):
    # Constructing the Encoder network using VGG16 with pretrained weights
    x = self.encoder(x)
    # Decoding the encoded feature maps using the Decoder network (DeConv Layers)
    x = self.decoder(x)
    return x



class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    
    self.conv_layers1 = nn.Sequential(OrderedDict([
                                            ('merge', nn.Conv2d(4, 3, kernel_size=1,stride = 1, padding = 0)),
                                            ('conv1', nn.Conv2d(3, 32, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu1', nn.ReLU(inplace=True)),
                                            ('pool1',  nn.MaxPool2d(4,4)),

                                            ('conv2_1', nn.Conv2d(32, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu2_1', nn.ReLU(inplace=True)),
                                            ('conv2_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu2_2', nn.ReLU(inplace=True)),
                                            ('pool2', nn.MaxPool2d(2,2)),

                                            ('conv3_1', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu3_1', nn.ReLU(inplace=True))

                                            ]))
    self.conv_layers2 = nn.Sequential(OrderedDict([
                                            # ('weight_norm3_1', nn.utils.weight_norm()), # Look into this later
                                            ('conv3_2', nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding = 1)),
                                            ('relu3_2', nn.ReLU(inplace=True)),
                                            # ('weight_norm3_2', nn.utils.weight_norm()), # Look into this later
                                            # ('pool3', nn.MaxPool2d(2,2))

                                            ]))

    self.fc_layers = nn.Sequential(OrderedDict([
                                            ('fc4', nn.Linear(12288, 60)),
                                            ('tanh4', nn.Tanh()),
                                            ('fc5', nn.Linear(100,2)),
                                            ('tanh5', nn.Tanh()),
                                            ('fc6', nn.Linear(2,1)),
                                            ('sigmoid', nn.Sigmoid())

                                                ]))

    self.maxpool = nn.MaxPool2d(2,2)

    # self.weight_norm1 = nn.utils.weight_norm(self.conv_layers1)
    # self.weight_norm2 = nn.utils.weight_norm(self.conv_layers2)
    # TODO : Look into adding weight norm layer in discriminator network
  def forward(self,x):
    x = self.conv_layers1(x)
    # x = nn.utils.weight_norm(x)
    x = self.conv_layers2(x)
    # x = nn.utils.weight_norm(x)
    # x = self.weight_norm1(x)
    # x = self.weight_norm2(x)
    x = self.maxpool(x)
    x = x.view(x.size(0),-1)
    x = self.fc_layers(x)

    return x
