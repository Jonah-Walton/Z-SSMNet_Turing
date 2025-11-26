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

import torch

from models.MNet_SegmentationNetwork import MNet

if __name__ == '__main__':
    MNet = MNet(1, 3, kn=(2, 2, 2, 2, 2), ds=True, FMU='sub')
    input = torch.randn((1, 1, 19, 255, 256))
    output = MNet(input)

    print([e.shape for e in output])

