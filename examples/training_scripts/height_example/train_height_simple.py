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
import logging
import time

from syne_tune import Reporter
from argparse import ArgumentParser

# To enable tuning, you have to report metrics from the training script
# so that they can be communicated later to FastBO.
# can be accomplished by calling report(epoch=, loss=).

# to only run the code when the program is run directly
# while the code is not executed when it is imported as a module
if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    args, _ = parser.parse_known_args()
    report = Reporter()

    for step in range(args.epochs):
        time.sleep(0.1)
        dummy_score = 1.0 / (0.1 + args.width * step / 100) + args.height * 0.1
        # Feed the score back to Syne Tune
        report(epoch=step + 1, mean_loss=dummy_score)
