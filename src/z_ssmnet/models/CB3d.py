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
from CNA3d import CNA3d

class CB3d(nn.Module): # conv block 3d
    def __init__(self, in_channels, out_channels, kSize=(3,3), stride=(1,1), padding=(1,1,1), bias=True,
                 norm_args:tuple=(None,None), activation_args:tuple=(None,None)):
        super().__init__()

        self.conv1 = CNA3d(in_channels, out_channels, kSize=kSize[0], stride=stride[0],
                             padding=padding, bias=bias, norm_args=norm_args[0], activation_args=activation_args[0])

        self.conv2 = CNA3d(out_channels, out_channels,kSize=kSize[1], stride=stride[1],
                             padding=padding, bias=bias, norm_args=norm_args[1], activation_args=activation_args[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x