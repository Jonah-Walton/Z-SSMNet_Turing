# Copyright 2022 Zhangfu Dong, Key Laboratory of Computer Network and Information Integration, Southeast University, Nanjing, China

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn
import torch
from nnunet.network_architecture.neural_network import SegmentationNetwork
from Down_BasicNet import Down
from Up_BasicNet import Up

class MNet(SegmentationNetwork):
    # ----------------------------------------parameters of NNUNet, not mine
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
    # ----------------------------------------

    def __init__(self, in_channels, num_classes, kn=(32, 48, 64, 80, 96), ds=True, FMU='sub'):
        """

        Args:
            in_channels: channels of input
            num_classes: output classes
            kn: the number of kernels
            ds: deep supervision
            FMU: type of feature merging unit
        """
        super().__init__()
        # ----------------------------------------parameters of NNUNet
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = ds
        self.num_classes = num_classes
        # ----------------------------------------


        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        fct = channel_factor[FMU]



        self.down11 = Down(in_channels, kn[0], ('/', 'both'), downsample=False)
        self.down12 = Down(kn[0], kn[1], ('2d', 'both'))
        self.down13 = Down(kn[1], kn[2], ('2d', 'both'))
        self.down14 = Down(kn[2], kn[3], ('2d', 'both'))
        self.bottleneck1 = Down(kn[3], kn[4], ('2d', '2d'))
        self.up11 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '2d'), FMU)
        self.up12 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '2d'), FMU)
        self.up13 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '2d'), FMU)
        self.up14 = Up(fct * (kn[0] + kn[1]), kn[0], ('both', 'both'), FMU)

        self.down21 = Down(kn[0], kn[1], ('3d', 'both'))
        self.down22 = Down(fct * kn[1], kn[2], ('both', 'both'), FMU)
        self.down23 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck2 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up21 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up22 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', 'both'), FMU)
        self.up23 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '3d'), FMU)

        self.down31 = Down(kn[1], kn[2], ('3d', 'both'))
        self.down32 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU)
        self.bottleneck3 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up31 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU)
        self.up32 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '3d'), FMU)

        self.down41 = Down(kn[2], kn[3], ('3d', 'both'), FMU)
        self.bottleneck4 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU)
        self.up41 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU)

        self.bottleneck5 = Down(kn[3], kn[4], ('3d', '3d'))

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )
        
    def forward(self, x):
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])

        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        bottleNeck5 = self.bottleneck5(down41[1])

        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])


        if self._deep_supervision and self.do_ds:
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            for i in range(7):
                features[i] = torch.sigmoid(features[i])
            return tuple([self.outputs[i](features[i]) for i in range(7)])
        else:
            return self.outputs[0](up14[0] + up14[1])
