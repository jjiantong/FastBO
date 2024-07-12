# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to implement FastBO
# Design new searcher FastBOSearcher and new methods for the searcher.
# Add a dataclass ``
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
from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass
from scipy.optimize import minimize

from syne_tune.optimizer.schedulers.searchers import GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.fastbo.curve_functions import (
    neg_log_likelihood,
    neg_log_likelihood_pow3,
    combined_function,
    pow3,
    log2,
)
from syne_tune.optimizer.schedulers.searchers.fastbo.hyperparameter import HyperparameterManager

logger = logging.getLogger(__name__)


@dataclass
class TrialObservations:
    trial_id: str
    num: List[int]
    metrics: List[float]


@dataclass
class TrialLearningCurveParameters:
    trial_id: str
    parameters: List[float]


class FastBOSearcher(GPFIFOSearcher):

    def __init__(
            self,
            config_space: Dict[str, Any],
            metric: str,
            points_to_evaluate: Optional[List[Dict[str, Any]]] = None,
            clone_from_state: bool = False,
            **kwargs,
    ):
        super().__init__(config_space, metric, points_to_evaluate, clone_from_state, **kwargs)
        self._max_resource_level = None

        self.trials_observations = []

        # Regarding performance decrease
        self.worse_1 = dict()
        self.worse_2 = dict()
        self._trials_to_compare = dict()

        self._hp = None

        # Estimated learning curve parameters for all trials
        self.trials_learning_curve_parameters = []

        # Two points for all trials
        self.trials_pre_exponential_points = dict()
        self.trials_saturation_points = dict()

    def create_hyperparameter_manager(self):
        self._hp = HyperparameterManager(dataset=self._dataset, benchmark=self._benchmark)

    def on_trial_result(
            self,
            trial_id: str,
            config: Dict[str, Any],
            result: Dict[str, Any],
            update: bool,
            record: Optional[bool] = False,
            estimate: Optional[bool] = False,
    ):
        """
        The difference from GPFIFOSearcher (ModelBasedSearcher) is the following
        two lines of code:
        In order to estimate trial_id's learning curve, we need to save each
        intermediate result in warm-up stage to `self.state_transformer._state`.
        """
        if record:
            self._save_result(trial_id, result)
        if estimate:
            self._estimate_learning_curve(trial_id, result)
        if update:
            self._update(trial_id, config, result)

    def _save_result(self, trial_id: str, result: Dict[str, Any]):
        """
        Saves the resource level and the corresponding metric result into
        `self.trials_observations` as the trial's a pair of observation.
        Meanwhile, decide whether the configuration encounters a continuous
        performance decrease.
        If it has observations already, new observation is appended. Otherwise,
        a new entry is appended.

        `self.trials_observations` is designed referring to
        `self.state_transformer._state.trials_evaluations`, but we don't want
        to add a new attribute in `self.state_transformer._state`, so just
        leave it here.
        """
        obs_y = result[self._metric]
        obs_x = result[self._resource_attr]
        # If mode is "min" (like loss or error), we first transfer it to the
        # metric like accuracy which is "max" mode
        if self._mode == "min":
            obs_y = (1 - obs_y) * 100

        # The following part is referred to syne_tune/optimizer/schedulers/
        # searchers/bayesopt/datatypes/tuning_job_state.py.
        # Specifically. `metrics_for_trial` and `_find_labeled` methods in
        # `TuningJobState` class.
        pos = self._find_observation(trial_id)
        if pos != -1:
            # The trial has observations in `self.trials_observations`
            # already. Append the new observation in `self.trials_observations`,
            # and compare it with the observation to compare.
            self.trials_observations[pos].metrics.append(obs_y)  # todo
            self.trials_observations[pos].num.append(obs_x)  # todo

            if obs_y < self._trials_to_compare[trial_id] * 0.99:  # todo: 0.99 & 0.9
                # If current value is worse than the value to be compared
                if self.worse_1[trial_id]:
                    # If we have observed a performance decrease already,
                    # set its `self.worse_2` to True to inform a continuous
                    # performance decrease
                    self.worse_2[trial_id] = True
                else:
                    # If it didn't have a performance decrease before, set
                    # its `self.worse_1` to True to inform the decrease
                    self.worse_1[trial_id] = True
            else:
                # If current value is equal or better than the value to be
                # compared, update the corresponding`self._trials_to_compare`
                # self.trials_observations[pos].metrics.append(obs_y)  # todo
                # self.trials_observations[pos].num.append(obs_x)  # todo
                self._trials_to_compare[trial_id] = obs_y
                if self.worse_1[trial_id]:
                    # If we have observed a performance decrease before, set
                    # its `self.worse_1` back to False as it didn't have the
                    # continuous performance decrease currently.
                    self.worse_1[trial_id] = False

        else:
            # Can't find this trial in `self.trials_observations`, which
            # means it is the first resource level of this trial. Add a
            # new entry in `self.trials_observations`. For the purpose of
            # detecting continuous performance decrease, (1) set
            # `self._trials_to_compare` to this first observation; and
            # (2) `self.worse_1` and `self.worse_2` are initialized to False
            metrics = [obs_y]
            num = [obs_x]
            new_obs = TrialObservations(trial_id=trial_id, num=num, metrics=metrics)
            self.trials_observations.append(new_obs)

            self._trials_to_compare[trial_id] = obs_y
            self.worse_1[trial_id] = False
            self.worse_2[trial_id] = False

    def _find_observation(self, trial_id: str) -> int:
        try:
            return next(
                i
                for i, x in enumerate(self.trials_observations)
                if x.trial_id == trial_id
            )
        except StopIteration:
            return -1

    def _estimate_learning_curve(self, trial_id: str, result: Dict[str, Any]):
        """
        Estimate the learning curve for `trial_id`.

        If the estimated parameters of the combined parametric model pass the
        validation test, save them as the last valid parameters and use them as
        the initial points of the next parameter estimation. Otherwise, try to
        use pow3 only. If the estimated parameters of pow3 still cannot pass the
        validation test, set pre-defined fixed values.
        """
        max_resource = self._max_resource_level
        resource = result[self._resource_attr]
        pos = self._find_observation(trial_id)

        # Before fitting the learning curve, we first check the distance between
        # min and max todo
        # max_value = max(self.trials_observations[pos].metrics)
        # min_value = min(self.trials_observations[pos].metrics)
        # dis = max_value - min_value
        # if dis < max_value * 0.01:
        #     self.trials_pre_exponential_points[trial_id] = resource
        #     self.trials_saturation_points[trial_id] = int(max_resource * 0.8)
        #     return

        # x = [i for i in range(1, resource + 1)]
        learning_curve_result = minimize(
                neg_log_likelihood,
                self._hp.valid_params_12.means(),
                method="TNC",
                args=(self.trials_observations[pos].num, self.trials_observations[pos].metrics),
            )

        parameters = learning_curve_result.x.tolist()
        if self._pass_simple_validation_test(params=parameters, num=12):
            # We got valid parameters for the combined model
            self._hp.valid_params_12.add_data(parameters)
            new_params = TrialLearningCurveParameters(trial_id=trial_id, parameters=parameters)
            self.trials_learning_curve_parameters.append(new_params)
            point1, point2 = self._estimate_two_points(parameters, resource, max_resource)
            self.trials_pre_exponential_points[trial_id] = point1
            self.trials_saturation_points[trial_id] = point2
        else:
            # We didn't get valid parameters for the combined model, try to fit
            # pow3 instead
            learning_curve_result = minimize(
                    neg_log_likelihood_pow3,
                    self._hp.valid_params_4.means(),
                    method="TNC",
                    args=(self.trials_observations[pos].num, self.trials_observations[pos].metrics)
                )
            parameters = learning_curve_result.x.tolist()
            if self._pass_simple_validation_test(params=parameters, num=4):
                # We got valid parameters for pow3
                self._hp.valid_params_4.add_data(parameters)
                new_params = TrialLearningCurveParameters(trial_id=trial_id, parameters=parameters)
                self.trials_learning_curve_parameters.append(new_params)
                point1, point2 = self._estimate_two_points(parameters, resource, max_resource)
                self.trials_pre_exponential_points[trial_id] = point1
                self.trials_saturation_points[trial_id] = point2
            else:
                # We didn't get valid parameters for pow3
                self.trials_pre_exponential_points[trial_id] = int(max_resource * 0.5)
                self.trials_saturation_points[trial_id] = int(max_resource * 0.8)

        # self.trials_saturation_points[trial_id] = int(max_resource)

    def _pass_simple_validation_test(self, params: List[float], num: int) -> bool:
        if num == 4:
            if params[0] < 0 or params[1] < 0 or params[2] < 0 or params[3] < 0:
                return False
        else:
            if (params[0] < 0 or params[0] > 1 or params[4] < 0 or params[4] > 1 or params[8] < 0 or params[8] > 1
                    or params[1] < 0 or params[2] < 0 or params[3] < 0
                    or params[5] < 0 or params[6] < 0 or params[7] < 0
                    or params[9] < 0 or params[10] > 0 or params[11] < 0):
                return False

        return True

    def _estimate_two_points(self, params: List[float], start_point: int, stop_point: int) -> tuple[int, int]:
        """
        Estimate pre-exponential point and saturation point given the estimated
        learning curve.

        Pre-exponential point: the first point anchor point x that satisfies
        f(bx) - f(x) < delta. Here we set b = 2 and delta = self._hp.t1 * f(x).

        Saturation point: all values after the anchor point x are in a distance
        less than a very small epsilon. Here we set epsilon = self._hp.t2 * f(x).
        We can compute the distance between f(x) and f(x1), where x1 is a far
        anchor point and here we use `stop_point`. To simplify the computation,
        use the limit of pow3 and exp3 (i.e., the parameter c) and only compute
        log2.
        """
        if len(params) == 12:
            func = combined_function
            far_value = func(params=params, x=stop_point)
        else:
            func = pow3
            far_value = params[0]

        pre_exponential_point = stop_point
        saturation_point = stop_point
        for i in range(start_point, stop_point):
            current_value = func(params=params, x=i)
            future_value = func(params=params, x=i * 2)
            if abs(future_value - current_value) < self._hp.t1 * current_value:
                pre_exponential_point = i
                break

        for i in range(pre_exponential_point, stop_point):
            current_value = func(params=params, x=i)
            if abs(far_value - current_value) < self._hp.t2 * current_value:
                saturation_point = i
                break

        # output = f"({pre_exponential_point}, {saturation_point})"
        # print(output)

        return pre_exponential_point, saturation_point
