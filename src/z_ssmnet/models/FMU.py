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

def FMU(x1, x2, mode='sub'):
    """
    feature merging unit
    Args:
        x1:
        x2:
        mode: types of fusion
    Returns:
    """
    if mode == 'sum':
        return torch.add(x1, x2)
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise Exception('Unexpected mode')