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


class FastBOManager:
    """
    Maintains two crucial points, called the pre-exponential point and
    the saturation point in the survey paper:
    Learning Curves for Decision Making in Supervised Machine Learning
    -- A Survey, by Felix Mohr, Jan N. van Rijn in 2022

    Mode "direct":
        Use the specified pre-exponential point and saturation point for
        each configuration. In this case, `self.warmup_point` is set as
        the specified pre-exponential point.
    Mode "indirect":
        Obtain pre-exponential point and saturation point for each con-
        figuration by modeling its learning curve, and use them. In this
        case, `self.pre_exponential_point` and `self.saturation_point`
        are no use.
    """

    def __init__(
            self,
            warmup_point: float,
            pre_exponential_point: float,
            saturation_point: float,
    ):
        if warmup_point == 1.0:
            self.mode = "direct"
            self.warmup_point = pre_exponential_point
            self.saturation_point = saturation_point
        else:
            self.mode = "indirect"
            self.warmup_point = warmup_point
