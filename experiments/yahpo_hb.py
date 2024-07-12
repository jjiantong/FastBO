# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to run experiments on YAHPO Gym
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
Running Hyperband on YAHPO Gym
Running sequentially among different 'benchmarks',
each with different datasets, using different number of workers,
limited with different max_num_trials_started
"""
import logging
from dataclasses import dataclass
import itertools
from tqdm import tqdm

from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import SyncHyperband
from syne_tune import Tuner, StoppingCriterion

# BenchmarkInfo is similar to SurrogateBenchmarkDefinition
# but be different especially for "max_t"
# either max_t or max_resource_attr should be used,
# in other benchmark, they all use max_resource_attr (e.g., "epochs")
# have tried max_resource_attr, worked but got different results
@dataclass
class BenchmarkInfo:
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name_list: str
    max_t: int
    resource_attr: str
    max_resource_attr: str


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)

    random_seed = 1

    benchmark_list = {
        "nb301": BenchmarkInfo(
            elapsed_time_attr="runtime",
            metric="val_accuracy",
            blackbox_name="yahpo-nb301",
            dataset_name_list=["CIFAR10"],
            mode="max",
            max_t=97,
            resource_attr="epoch",
            max_resource_attr="epochs",
        ),
        # Some notes on available datasets for yahpo-lcbench:
        # From benchmarking.commons.benchmark_definitions.yahpo, we can see
        # the available datasets are loaded from openml_task_name_to_id.json,
        # there is also a lcbench_selected_datasets that includes 5 datasets.
        # The original example here uses "3945", the value in the json file,
        # which corresponds to the id "KDDCup09_appetency". So here I just
        # found the 5 values related to the 5 selected datasets, which are
        # "189908", "189354", "189866", "7593", "168908", so added them
        "lcbench": BenchmarkInfo(
            elapsed_time_attr="time",
            metric="val_accuracy",
            blackbox_name="yahpo-lcbench",
            dataset_name_list=["3945", "189908", "189354", "189866", "7593", "168908"],
            mode="max",
            max_t=51,
            resource_attr="epoch",
            max_resource_attr="epochs",
        ),
        # Some notes on yahpo-fcnet:
        # There is no fcnet scenario mentioned in the YAHPO Gym paper, and also
        # no yahpo-fcnet included in benchmarking.commons.benchmark_definitions
        # (yahpo.py or fcnet.py). There is no "fcnet_naval_propulsion" in fcnet.py
        "fcnet": BenchmarkInfo(
            elapsed_time_attr="runtime",
            metric="valid_mse",
            blackbox_name="yahpo-fcnet",
            dataset_name_list=["fcnet_naval_propulsion"],
            mode="min",
            max_t=99,
            resource_attr="epoch",
            max_resource_attr="epochs",
        ),
    }

    n_workers_list = [4]
    max_num_trials_started_list = [100]

    for benchmark in ["nb301", "lcbench", "fcnet"]:
        benchmark_info = benchmark_list[benchmark]

        combinations = list(
            itertools.product(benchmark_info.dataset_name_list, n_workers_list, max_num_trials_started_list)
        )

        for dataset_name, n_workers, max_num_trials_started in tqdm(combinations):
            print("\nbenchmark =", benchmark, ", dataset =", dataset_name, ", n_workers =", n_workers)

            max_resource_attr = benchmark_info.max_resource_attr
            trial_backend = BlackboxRepositoryBackend(
                elapsed_time_attr=benchmark_info.elapsed_time_attr,
                blackbox_name=benchmark_info.blackbox_name,
                dataset=dataset_name,
            )

            config_space = dict(
                trial_backend.blackbox.configuration_space,
                **{max_resource_attr: benchmark_info.max_t},
            )
            blackbox = trial_backend.blackbox
            scheduler = SyncHyperband(
                config_space=config_space,
                max_resource_attr=max_resource_attr,
                resource_attr=benchmark_info.resource_attr,
                mode=benchmark_info.mode,
                metric=benchmark_info.metric,
                random_seed=random_seed,
            )

            stop_criterion = StoppingCriterion(max_num_trials_started=max_num_trials_started)

            # It is important to set ``sleep_time`` to 0 here (mandatory for simulator
            # backend)
            tuner = Tuner(
                trial_backend=trial_backend,
                scheduler=scheduler,
                stop_criterion=stop_criterion,
                n_workers=n_workers,
                sleep_time=0,
                print_update_interval=10,
                # This callback is required in order to make things work with the
                # simulator callback. It makes sure that results are stored with
                # simulated time (rather than real time), and that the time_keeper
                # is advanced properly whenever the tuner loop sleeps
                callbacks=[SimulatorCallback()],
                tuner_name=f"ASHA-Yahpo-{benchmark}",
            )
            tuner.run()