# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to run experiments on FCNet
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
"""
Running BOHB on FCNet
Running sequentially among different datasets, using different number of workers,
limited with different max_wallclock_time
"""
import logging
import itertools
from tqdm import tqdm

from benchmarking.commons.benchmark_definitions.fcnet import fcnet_benchmark
from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import SyncBOHB
from syne_tune import Tuner, StoppingCriterion


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)

    random_seed = 70
    fcnet_random_seed = 0
    # to run experiments using different datasets, #workers, max wallclock time
    dataset_name_list = ["slice_localization"]
    # dataset_name_list = ["protein_structure", "naval_propulsion", "parkinsons_telemonitoring", "slice_localization"]
    n_workers_list = [4]
    max_wallclock_time_list = [
        # 10, 60, 360, 600, 1000, 1200, 1500, 1800, 3000,
        3600, 2 * 3600,
        3 * 3600, 4 * 3600,
        5 * 3600, 6 * 3600,
    ]
    # n_workers_list = [1, 2, 4]
    # max_wallclock_time_list = [3600]
    # max_num_trials_started_list = [256, 512, 1024]
    combinations = list(
        itertools.product(dataset_name_list, n_workers_list, max_wallclock_time_list)
    )
    print(combinations)

    for dataset_name, n_workers, max_wallclock_time in tqdm(combinations):
        print("\nFCNet,", dataset_name, ", n_workers =", n_workers, ", max_wallclock_time =", max_wallclock_time)

        benchmark = fcnet_benchmark(dataset_name)

        # Simulator backend specialized to tabulated blackboxes
        max_resource_attr = benchmark.max_resource_attr
        trial_backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            max_resource_attr=max_resource_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=dataset_name,
            seed=fcnet_random_seed,
        )

        # The original BOHB is synchronous
        blackbox = trial_backend.blackbox
        # restrict_configurations = blackbox.all_configurations()
        scheduler = SyncBOHB(
            config_space=blackbox.configuration_space_with_max_resource_attr(
                max_resource_attr
            ),
            max_resource_attr=max_resource_attr,
            resource_attr=blackbox.fidelity_name(),
            mode=benchmark.mode,
            metric=benchmark.metric,
            random_seed=random_seed,
            # search_options=dict(restrict_configurations=restrict_configurations),
        )

        stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
        # stop_criterion = StoppingCriterion(max_num_trials_started=60000)

        # Printing the status during tuning takes a lot of time, and so does
        # storing results.
        print_update_interval = 700
        results_update_interval = 300
        # It is important to set ``sleep_time`` to 0 here (mandatory for simulator
        # backend)
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            sleep_time=0,
            results_update_interval=results_update_interval,
            print_update_interval=print_update_interval,
            # This callback is required in order to make things work with the
            # simulator callback. It makes sure that results are stored with
            # simulated time (rather than real time), and that the time_keeper
            # is advanced properly whenever the tuner loop sleeps
            callbacks=[SimulatorCallback()],
        )
        tuner.run()


