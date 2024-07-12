# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to implement FastBO
# Design new scheduler FastBOScheduler and new methods for the scheduler
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
from typing import Dict, Any, Optional, Set
import logging

from syne_tune.backend.time_keeper import RealTimeKeeper
from syne_tune.config_space import cast_config_values
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.multi_fidelity import MultiFidelitySchedulerMixin
from syne_tune.optimizer.schedulers.fastbo.fastbo_manager import (
    FastBOManager,
)
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments import (
    filter_by_key,
)
from syne_tune.optimizer.scheduler import (
    SchedulerDecision, TrialSuggestion,
)
from syne_tune.backend.trial_status import Trial

logger = logging.getLogger(__name__)

_ARGUMENT_KEYS = {
    # The kwargs that will not pass to the constructor of superclass (FIFOScheduler)
    # In other word, FIFOScheduler doesn't deal with these kwargs
    "max_resource_attr",
    "resource_attr",
    "benchmark",
    "dataset",
    "warmup_point",
    "pre_exponential_point",
    "saturation_point",
}


class FastBOScheduler(FIFOScheduler, MultiFidelitySchedulerMixin):

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        super().__init__(config_space, **filter_by_key(kwargs, _ARGUMENT_KEYS))
        self._create_internal(**kwargs)

    def _create_internal(self, **kwargs):
        self.max_resource_attr = kwargs.get("max_resource_attr")
        self._resource_attr = kwargs["resource_attr"]
        if self.max_resource_attr is None:
            logger.warning(
                "You do not specify max_resource_attr, but use max_resource_level "
                "instead. This is not recommended best practice and may lead to a "
                "loss of efficiency. Consider using max_resource_attr instead.\n"
                "See https://syne-tune.readthedocs.io/en/latest/tutorials/multifidelity/mf_setup.html#the-launcher-script "
                "for details."
            )
        self._max_resource_level = self._infer_max_resource_level(
            max_resource_level=kwargs.get("max_resource_level"),
            max_resource_attr=self.max_resource_attr,
        )
        assert self._max_resource_level is not None, (
            "Maximum resource level has to be specified, please provide "
            "max_resource_attr or max_resource_level argument."
        )

        self._searcher._resource_attr = self._resource_attr
        self._searcher._max_resource_level = self._max_resource_level

        self._searcher._benchmark = kwargs.get("benchmark")
        self._searcher._dataset = kwargs.get("dataset")
        self.searcher.create_hyperparameter_manager()

        # Creates `FastBOManager`.
        # Case 1: warmup point is specified.
        #       Pre-exponential point and saturation point are obtained by learning
        #       curve modeling. If pre-exponential point or saturation point is also
        #       specified here, ignore it.
        # Case 2: warmup point is not specified, pre-exponential point is specified.
        #       Do partial evaluation for each configuration according to this pre-
        #       exponential point. If saturation point is specified, use it in post-
        #       processing; but if not, just do a full evaluation in post-processing.
        # Case 3: warmup point and pre-exponential point are not specified.
        #       Run a traditional BO.
        # If warmup point, pre-exponential point, or saturation point is specified by
        # the number rather than the ratio, transfer it to ratio. TODO: haven't

        # default values
        warmup_point = 1.0
        pre_exponential_point = 1.0
        saturation_point = 1.0
        if 'warmup_point' in kwargs:
            # pre-exponential point and saturation point will be ignored in this case
            warmup_point = kwargs.get("warmup_point")
        else:
            if 'pre_exponential_point' in kwargs:
                pre_exponential_point = kwargs.get("pre_exponential_point")
            if 'saturation_point' in kwargs:
                saturation_point = kwargs.get("saturation_point")

        self.fastbo_manager = FastBOManager(
            warmup_point=warmup_point,
            pre_exponential_point=pre_exponential_point,
            saturation_point=saturation_point,
        )

        # Maps trial_id to config
        self._trial_to_config = dict()

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        # If no time keeper was provided at construction, we use a local
        # one which is started here
        if self.time_keeper is None:
            self.time_keeper = RealTimeKeeper()
            self.time_keeper.start_of_time()
        # For pause/resume schedulers: Can a paused trial be promoted?
        promote_trial_id, extra_kwargs = self._promote_trial(new_trial_id=str(trial_id))
        if promote_trial_id is not None:
            return TrialSuggestion.resume_suggestion(
                trial_id=int(promote_trial_id), config=extra_kwargs
            )
        # Ask searcher for config of new trial to start
        extra_kwargs["elapsed_time"] = self._elapsed_time()
        str_trial_id = str(trial_id)
        config = self.searcher.get_config(**extra_kwargs, trial_id=str_trial_id)
        if config is not None:
            config = cast_config_values(config, self.config_space)
            # The only difference from `FIFOScheduler`: adding the following
            # three lines of code. We need to save the config of a new trial
            # into `_trial_to_config` for resuming it in the future. The saved
            # config should include `max_resource_attr`.
            if self.max_resource_attr is not None:
                config2 = dict(config, **{self.max_resource_attr: self._max_resource_level})
            self._trial_to_config[trial_id] = config2
            config = self._on_config_suggest(config, str_trial_id, **extra_kwargs)
            config = TrialSuggestion.start_suggestion(config)
        return config

    def _suggest_final_resumed_task(self, trial_id: int) -> Optional[TrialSuggestion]:
        """Implements ``suggest_final_resumed_task``, except for basic
        postprocessing of config.

        We pause a paused trial.

        :param trial_id: ID for a paused trial to be resumed
        :return: Suggestion for a trial to be resumed. If no suggestion can be
        made, None is returned
        """
        # Paused trial to be resumed
        _config = self._trial_to_config[trial_id]
        str_trial_id = str(trial_id)
        if self.max_resource_attr is not None and str_trial_id in self.searcher.trials_saturation_points:
            config = dict(_config,
                          **{self.max_resource_attr: self.searcher.trials_saturation_points[str_trial_id]})
        else:
            config = _config
        suggestion = TrialSuggestion.resume_suggestion(
            trial_id=trial_id, config=config
        )
        return suggestion

    def on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """
        We simply relay ``result`` to the searcher. Other decisions are done
        in ``on_trial_complete``.
        """
        self._check_result(result)
        trial_id = str(trial.trial_id)
        config = self._preprocess_config(trial.config)

        max_resource = self._max_resource_level
        resource = int(result[self._resource_attr])
        warm_up = int(max_resource * self.fastbo_manager.warmup_point)
        if resource < warm_up:
            # Stage 1: before warmup point: record the result. If it has
            # encountered a continuous performance decrease, pause and update,
            # which means to skip the learning curve module and directly set
            # `self.searcher.trials_pre_exponential_points[trial_id]` to the
            # current position. Otherwise, continue.
            if self.fastbo_manager.mode == "direct":
                trial_decision = SchedulerDecision.CONTINUE
                self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=False,
                                              estimate=False)  # note: do nothing here
            else:
                self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=True, estimate=False)
                if self.searcher.worse_2[trial_id]:
                    trial_decision = SchedulerDecision.PAUSE
                    self.searcher.trials_pre_exponential_points[trial_id] = resource
                    self.searcher.trials_saturation_points[trial_id] = max_resource
                    self.searcher.on_trial_result(trial_id, config, result=result,
                                                  update=True, record=False, estimate=False)
                else:
                    trial_decision = SchedulerDecision.CONTINUE
        elif resource == warm_up:
            # Stage 2: warmup point: record the result, estimate two points.
            # If the estimated pre-exponential point is just the warmup point,
            # pause and update. Otherwise, continue.
            if self.fastbo_manager.mode == "direct":
                # Case 1: "direct" mode. We have reached the pre-exponential
                # point (which is set on `warmup_point`), so pause and update.
                # Directly use the specified points.
                trial_decision = SchedulerDecision.PAUSE
                self.searcher.on_trial_result(trial_id, config, result=result, update=True, record=True, estimate=False)
                self.searcher.trials_pre_exponential_points[trial_id] = warm_up
                self.searcher.trials_saturation_points[trial_id] = \
                    int(max_resource * self.fastbo_manager.saturation_point)
            else:
                # Case 2: "indirect" mode. We need to estimate learning curve
                # to estimate two points. After obtaining the estimated points,
                # we need to decide if the pre-exponential point is the warmup
                # point.
                self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=True, estimate=True)
                if self.searcher.trials_pre_exponential_points[trial_id] - warm_up == 0:
                    trial_decision = SchedulerDecision.PAUSE
                    self.searcher.on_trial_result(trial_id, config, result=result, update=True, record=False,
                                                  estimate=False)
                else:
                    trial_decision = SchedulerDecision.CONTINUE
                    self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=False,
                                                  estimate=False)  # note: do nothing here
        elif resource < self.searcher.trials_pre_exponential_points[trial_id]:
            # Stage 3: after warmup point, before pre-exponential point: continue
            trial_decision = SchedulerDecision.CONTINUE
            self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=False, estimate=False)
        elif resource == self.searcher.trials_pre_exponential_points[trial_id]:
            # Stage 4: pre-exponential point: pause and update
            trial_decision = SchedulerDecision.PAUSE
            self.searcher.on_trial_result(trial_id, config, result=result, update=True, record=False, estimate=False)

        # Extra info in debug mode
        log_msg = f"trial_id {trial_id} ("
        if self.is_multiobjective_scheduler():
            metrics = {k: result[k] for k in self.metric}
        else:
            metrics = {"metric": result[self.metric]}
        log_msg += ", ".join([f"{k} = {v:.3f}" for k, v in metrics.items()])
        for k, is_float in (("epoch", False), ("elapsed_time", True)):
            if k in result:
                if is_float:
                    log_msg += f", {k} = {result[k]:.2f}"
                else:
                    log_msg += f", {k} = {result[k]}"
        log_msg += f"): decision = {trial_decision}"
        logger.debug(log_msg)

        return trial_decision

    def final_on_trial_result(self, trial: Trial, result: Dict[str, Any]) -> str:
        """
        We simply relay ``result`` to the searcher. Other decisions are done
        in ``on_trial_complete``.

        This method is called only when handling the final remaining tasks.
        It is different from `on_trial_result` because we need to ensure a
        `CONTINUE` return each time, without considering the pre-exponential
        point.
        """
        self._check_result(result)
        trial_id = str(trial.trial_id)
        config = self._preprocess_config(trial.config)

        trial_decision = SchedulerDecision.CONTINUE
        self.searcher.on_trial_result(trial_id, config, result=result, update=False, record=False, estimate=False)

        # Extra info in debug mode
        log_msg = f"trial_id {trial_id} ("
        if self.is_multiobjective_scheduler():
            metrics = {k: result[k] for k in self.metric}
        else:
            metrics = {"metric": result[self.metric]}
        log_msg += ", ".join([f"{k} = {v:.3f}" for k, v in metrics.items()])
        for k, is_float in (("epoch", False), ("elapsed_time", True)):
            if k in result:
                if is_float:
                    log_msg += f", {k} = {result[k]:.2f}"
                else:
                    log_msg += f", {k} = {result[k]}"
        log_msg += f"): decision = {trial_decision}"
        logger.debug(log_msg)
        return trial_decision

    def set_default_point_values(self, trials_ids: Set[int]):
        max_resource = self._max_resource_level
        for trial_id in trials_ids:
            self.searcher.trials_pre_exponential_points[str(trial_id)] = max_resource * 0.5
            self.searcher.trials_saturation_points[str(trial_id)] = max_resource * 0.8

