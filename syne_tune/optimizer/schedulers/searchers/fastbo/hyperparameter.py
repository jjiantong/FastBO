# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to implement FastBO
# Create the FastBOManager class as a helper class for FastBO's scheduler
#
# --- Original Apache License and Copyright ---
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import List


class InitialParameter:
    def __init__(self, num: int):
        if num == 12:
            self.data = [[] for _ in range(12)]
        else:
            self.data = [[] for _ in range(4)]

    def add_data(self, new_data: List[float]):
        for i, value in enumerate(new_data):
            self.data[i].append(value)

    def means(self) -> List[float]:
        means = [sum(values) / len(values) if values else 0 for values in self.data]
        return means


class HyperparameterManager:

    def __init__(self, dataset, benchmark):
        self._dataset = dataset
        self._benchmark = benchmark
        initial_params_12 = self.load_initial_params_12()
        initial_params_4 = self.load_initial_params_4()

        # Save the last valid estimated parameters that can be used for the
        # initial points of the next estimation
        self.valid_params_12 = InitialParameter(num=12)
        self.valid_params_12.add_data(initial_params_12)
        self.valid_params_4 = InitialParameter(num=4)
        self.valid_params_4.add_data(initial_params_4)

        # Two thresholds, related to the pre-exponential point and the
        # saturation point
        self.t1 = 0.002
        self.t2 = 0.0005
        if self._benchmark == "fcnet":
            if self._dataset == "protein_structure":
                self.t1 = 0.06
                self.t2 = 0.001
            elif self._dataset == "slice_localization":
                self.t1 = 0.002
                self.t2 = 0.0002

    def load_initial_params_12(self):
        if self._benchmark == "lcbench":
            if self._dataset == "airlines":
                return [0.333, 59, 6, 1.5,
                        0.333, 59, 1.5, 4,
                        0.333, 59, -0.134,
                        0.1]
            elif self._dataset == "Fashion-MNIST":
                return [0.333, 63, 6, 1.5,
                        0.333, 63, 1.5, 4,
                        0.333, 63, -0.134,
                        0.1]
            elif self._dataset == "albert":
                return [0.333, 58, 6, 1.5,
                        0.333, 58, 1.5, 4,
                        0.333, 58, -0.134,
                        0.1]
            elif self._dataset == "covertype":
                return [0.333, 54, 6, 1.5,
                        0.333, 54, 1.5, 4,
                        0.333, 54, -0.134,
                        0.1]
            elif self._dataset == "christine":
                return [0.333, 68, 6, 1.5,
                        0.333, 68, 1.5, 4,
                        0.333, 68, -0.134,
                        0.1]
        elif self._benchmark == "fcnet":
            if self._dataset == "protein_structure":
                return [0.333, 57, 6, 1.5,
                        0.333, 57, 1.5, 4,
                        0.333, 57, -0.134,
                        0.1]
            elif self._dataset == "slice_localization":
                return [0.333, 90, 6, 1.5,
                        0.333, 90, 1.5, 4,
                        0.333, 90, -0.134,
                        0.1]
        elif self._benchmark == "nb201":
            if self._dataset == "cifar10":
                return [0.333, 70, 6, 1.5,
                        0.333, 70, 1.5, 4,
                        0.333, 70, -0.134,
                        0.1]
            elif self._dataset == "cifar100":
                return [0.333, 56, 6, 1.5,
                        0.333, 56, 1.5, 4,
                        0.333, 56, -0.134,
                        0.1]
            elif self._dataset == "ImageNet16-120":
                return [0.333, 40, 6, 1.5,
                        0.333, 40, 1.5, 4,
                        0.333, 40, -0.134,
                        0.1]

        # For all the other cases
        return [0.333, 90, 6, 1.5,
                0.333, 90, 1.5, 4,
                0.333, 90, -0.134,
                0.1]

    def load_initial_params_4(self):
        if self._benchmark == "lcbench":
            if self._dataset == "airlines":
                return [59, 6, 1.712, 0.1]
            elif self._dataset == "Fashion-MNIST":
                return [73, 6, 1.712, 0.1]
            elif self._dataset == "albert":
                return [58, 6, 1.5, 0.1]
            elif self._dataset == "covertype":
                return [54, 6, 1.5, 0.1]
            elif self._dataset == "christine":
                return [68, 6, 1.712, 0.1]
        elif self._benchmark == "fcnet":
            if self._dataset == "protein_structure":
                return [57, 6, 1.5, 0.01]
            elif self._dataset == "slice_localization":
                return [90, 6, 1.5, 0.01]
        elif self._benchmark == "nb201":
            if self._dataset == "cifar10":
                return [71, 6, 1.5, 0.1]
            elif self._dataset == "cifar100":
                return [56, 6, 1.5, 0.1]
            elif self._dataset == "ImageNet16-120":
                return [40, 6, 1.5, 0.01]

        # For all the other cases
        return [90, 6, 1.5, 0.01]
