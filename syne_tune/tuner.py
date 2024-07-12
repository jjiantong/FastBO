# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is modified by Jiantong Jiang to implement FastBO.
# Add extra steps after the main loop of `run` for FastBO, to finish full
# evaluations for the top-n best-performing trials
# Design new methods, including:
#   - `_fastbo_postprocess`
#   - `_pause_running_trials`
#   - `_schedule_final_tasks`
# Conduct major modifications on the following methods:
#   - `run`
#   - `_process_new_results`
#   - `_update_running_trials`
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
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import dill as dill

from syne_tune.backend.simulator_backend.events import OnTrialResultEvent
from syne_tune.backend.trial_backend import (
    TrialBackend,
    TrialAndStatusInformation,
    TrialIdAndResultList,
)
from syne_tune.backend.trial_status import Status, Trial, TrialResult
from syne_tune.callbacks.remove_checkpoints_callback import RemoveCheckpointsCallback
from syne_tune.constants import (
    ST_TUNER_CREATION_TIMESTAMP,
    ST_TUNER_START_TIMESTAMP,
    ST_METADATA_FILENAME,
    ST_TUNER_DILL_FILENAME,
)
from syne_tune.optimizer.scheduler import SchedulerDecision, TrialScheduler
from syne_tune.optimizer.fastbo import FastBO
from syne_tune.tuner_callback import TunerCallback
from syne_tune.results_callback import StoreResultsCallback
from syne_tune.tuning_status import TuningStatus, print_best_metric_found, find_best_n_trials_ids
from syne_tune.util import (
    RegularCallback,
    check_valid_sagemaker_name,
    experiment_path,
    name_from_base,
    dump_json_with_numpy,
)

logger = logging.getLogger(__name__)

DEFAULT_SLEEP_TIME = 5.0


class Tuner:
    """
    Controller of tuning loop, manages interplay between scheduler and
    trial backend. Also, stopping criterion and number of workers are
    maintained here.

    :param trial_backend: Backend for trial evaluations
    :param scheduler: Tuning algorithm for making decisions about which
        trials to start, stop, pause, or resume
    :param stop_criterion: Tuning stops when this predicates returns ``True``.
        Called in each iteration with the current tuning status. It is
        recommended to use :class:`StoppingCriterion`.
    :param n_workers: Number of workers used here. Note that the backend
        needs to support (at least) this number of workers to be run
        in parallel
    :param sleep_time: Time to sleep when all workers are busy. Defaults to
        :const:`DEFAULT_SLEEP_TIME`
    :param results_update_interval: Frequency at which results are updated and
        stored (in seconds). Defaults to 10.
    :param print_update_interval: Frequency at which result table is printed.
        Defaults to 30.
    :param max_failures: This many trial execution failures are allowed before
        the tuning loop is aborted. Defaults to 1
    :param tuner_name: Name associated with the tuning experiment, default to
        the name of the entrypoint. Must consists of alpha-digits characters,
        possibly separated by '-'. A postfix with a date time-stamp is added
        to ensure uniqueness.
    :param asynchronous_scheduling: Whether to use asynchronous scheduling
        when scheduling new trials. If ``True``, trials are scheduled as soon as
        a worker is available. If ``False``, the tuner waits that all trials
        are finished before scheduling a new batch of size ``n_workers``.
        Default to ``True``.
    :param wait_trial_completion_when_stopping: How to deal with running
        trials when stopping criterion is met. If ``True``, the tuner waits
        until all trials are finished. If ``False``, all trials are terminated.
        Defaults to ``False``.
    :param callbacks: Called at certain times in the tuning loop, for example
        when a result is seen. The default callback stores results every
        ``results_update_interval``.
    :param metadata: Dictionary of user-metadata that will be persisted in
        ``{tuner_path}/{ST_METADATA_FILENAME}``, in addition to metadata provided by
        the user. ``SMT_TUNER_CREATION_TIMESTAMP`` is always included which
        measures the time-stamp when the tuner started to run.
    :param suffix_tuner_name: If ``True``, a timestamp is appended to the
        provided ``tuner_name`` that ensures uniqueness, otherwise the name is
        left unchanged and is expected to be unique. Defaults to ``True``.
    :param save_tuner: If ``True``, the :class:`Tuner` object is serialized at
        the end of tuning, including its dependencies (e.g., scheduler). This
        allows all details of the experiment to be recovered. Defaults to
        ``True``.
    :param start_jobs_without_delay: Defaults to ``True``. If this is ``True``, the tuner
        starts new jobs depending on scheduler decisions communicated to the
        backend. For example, if a trial has just been stopped (by calling
        ``backend.stop_trial``), the tuner may start a new one immediately, even
        if the SageMaker training job is still busy due to stopping delays.
        This can lead to faster experiment runtime, because the backend is
        temporarily going over its budget.

        If set to ``False``, the tuner always asks the backend for the number of
        busy workers, which guarantees that we never go over the ``n_workers``
        budget. This makes a difference for backends where stopping or pausing
        trials is not immediate (e.g., :class:`SageMakerBackend`). Not going
        over budget means that ``n_workers`` can be set up to the available quota,
        without running the risk of an exception due to the quota being
        exceeded. If you get such exceptions, we recommend to use
        ``start_jobs_without_delay=False``. Also, if the SageMaker warm pool
        feature is used, it is recommended to set
        ``start_jobs_without_delay=False``, since otherwise more than ``n_workers``
        warm pools will be started, because existing ones are busy with
        stopping when they should be reassigned.
    :param trial_backend_path: If this is given, the path of ``trial_backend``
        (where logs and checkpoints of trials are stored) is set to this.
        Otherwise, it is set to ``self.tuner_path``, so that per-trial
        information is written to the same path as tuning results.

        If the backend is :class:`~syne_tune.backend.LocalBackend` and the
        experiment is ru remotely, we recommend to set this, since otherwise
        checkpoints and logs are synced to S3, along with tuning results, which
        is costly and error-prone.
    :param early_removal_checkpoints: If ``True``, we remove checkpoints of
        certain paused trials early, if this is supported by the scheduler, see
        :meth:`syne_tune.optimizer.scheduler.TrialScheduler.trials_checkpoints_can_be_removed`.
        Defaults to ``True``.
    """

    def __init__(
        self,
        trial_backend: TrialBackend,
        scheduler: TrialScheduler,
        stop_criterion: Callable[[TuningStatus], bool],
        n_workers: int,
        sleep_time: float = DEFAULT_SLEEP_TIME,
        results_update_interval: float = 10.0,
        print_update_interval: float = 30.0,
        max_failures: int = 1,
        tuner_name: Optional[str] = None,
        asynchronous_scheduling: bool = True,
        wait_trial_completion_when_stopping: bool = False,
        callbacks: Optional[List[TunerCallback]] = None,
        metadata: Optional[dict] = None,
        suffix_tuner_name: bool = True,
        save_tuner: bool = True,
        start_jobs_without_delay: bool = True,
        trial_backend_path: Optional[str] = None,
        early_removal_checkpoints: bool = True,
    ):
        self.trial_backend = trial_backend
        self.scheduler = scheduler
        self.n_workers = n_workers
        self.sleep_time = sleep_time
        self.results_update_interval = results_update_interval
        self.stop_criterion = stop_criterion
        self.asynchronous_scheduling = asynchronous_scheduling
        self.wait_trial_completion_when_stopping = wait_trial_completion_when_stopping
        self.metadata = self._enrich_metadata(metadata)
        self.save_tuner = save_tuner
        self.start_jobs_without_delay = start_jobs_without_delay

        self.max_failures = max_failures
        self.print_update_interval = print_update_interval

        if tuner_name is not None:
            check_valid_sagemaker_name(tuner_name)
        else:
            tuner_name = Path(self.trial_backend.entrypoint_path()).stem.replace(
                "_", "-"
            )
        if suffix_tuner_name or tuner_name is None:
            self.name = name_from_base(tuner_name, default="st-tuner")
        else:
            self.name = tuner_name

        # we keep track of the last result seen to send it to schedulers when trials complete.
        self.last_seen_result_per_trial = {}
        self.trials_scheduler_stopped = set()
        self.tuner_path = Path(experiment_path(tuner_name=self.name))

        # inform the backend to the folder of the Tuner. This allows the local backend
        # to store the logs and tuner results in the same folder.
        self.trial_backend.set_path(
            results_root=str(self.tuner_path)
            if trial_backend_path is None
            else trial_backend_path,
            tuner_name=self.name,
        )
        self._init_callbacks(callbacks, early_removal_checkpoints)
        self.tuning_status = None
        self.tuner_saver = None
        self.status_printer = None

    def _init_callbacks(
        self, callbacks: Optional[List[TunerCallback]], early_removal_checkpoints: bool
    ):
        if callbacks is None:
            callbacks = [self._default_callback()]
        else:
            if not any(
                isinstance(callback, StoreResultsCallback) for callback in callbacks
            ):
                logger.warning(
                    "None of the callbacks provided are of type StoreResultsCallback. "
                    "This means no tuning results will be written."
                )
        if early_removal_checkpoints:
            callbacks.append(RemoveCheckpointsCallback())
        self.callbacks = callbacks

    def run(self):
        """Launches the tuning."""
        done_trials_statuses = OrderedDict()
        try:
            logger.info(f"results of trials will be saved on {self.tuner_path}")

            if self.tuning_status is None:
                # time counter started
                self.tuning_status = TuningStatus(
                    metric_names=self.scheduler.metric_names()
                )
            # prints the status every ``print_update_interval`` seconds
            self.status_printer = RegularCallback(
                callback=lambda tuning_status: logger.info(
                    "tuning status (last metric is reported)\n" + str(tuning_status)
                ),
                call_seconds_frequency=self.print_update_interval,
            )
            # saves the tuner every ``results_update_interval`` seconds
            if self.save_tuner:
                self.tuner_saver = RegularCallback(
                    callback=lambda tuner: tuner.save(),
                    call_seconds_frequency=self.results_update_interval,
                )

            self.metadata[ST_TUNER_START_TIMESTAMP] = time.time()

            for callback in self.callbacks:
                callback.on_tuning_start(self)

            self.tuner_path.mkdir(exist_ok=True, parents=True)

            self._save_metadata()

            # ``running_trial_ids`` contains the ids of all trials currently running,
            # whether they were started from scratch or were resumed from a pausing
            # state
            running_trials_ids = set()

            config_space_exhausted = False
            stop_condition_reached = self._stop_condition()

            while (
                # we stop when either the stop condition is reached
                not stop_condition_reached
                # or when all trials are done if the wait_trial_completion is activated
                or self.wait_trial_completion_when_stopping
                and len(running_trials_ids) > 0
            ):
                for callback in self.callbacks:
                    callback.on_loop_start()

                new_done_trial_statuses, new_results = self._process_new_results(
                    running_trials_ids=running_trials_ids,
                )

                if new_results and self.save_tuner:
                    # Save tuner state only if there have been new results
                    self.tuner_saver(tuner=self)

                # update the list of done trials and remove those from ``running_trials_ids``
                # Note: It is important to update ``running_trials_ids`` before
                # calling ``_schedule_new_tasks``.
                # Otherwise, a trial can be registered as paused in
                # ``_process_new_results``, and immediately be resumed in
                # ``_schedule_new_tasks``. If ``new_done_trial_statuses`` is subtracted from
                # ``running_trials_ids`` afterwards only, this trial is removed from
                # ``running_trials_ids`` even though it is running. Also, its status remains
                # paused, because the next call of ``_process_new_results`` only considers
                # trials in ``running_trials_ids``.
                done_trials_statuses.update(new_done_trial_statuses)
                running_trials_ids.difference_update(new_done_trial_statuses.keys())

                if (
                    config_space_exhausted
                    or self.wait_trial_completion_when_stopping
                    and stop_condition_reached
                ):
                    # if the search space is exhausted, we loop until the running trials are done or until the
                    # stop condition is reached
                    if len(running_trials_ids) > 0:
                        if config_space_exhausted:
                            logger.debug(
                                f"Configuration space exhausted, waiting for completion of running trials "
                                f"{running_trials_ids}"
                            )
                        else:
                            logger.debug(
                                f"Stopping criterion reached, waiting for completion of running trials "
                                f"{running_trials_ids}"
                            )
                        self._sleep()
                    else:
                        break
                else:
                    try:
                        self._schedule_new_tasks(running_trials_ids=running_trials_ids)
                    except StopIteration:
                        logger.info(
                            "Tuning is finishing as the whole configuration space got exhausted."
                        )
                        config_space_exhausted = True
                        print(
                            "Tuning is finishing as the whole configuration space got exhausted."
                        )

                self.status_printer(self.tuning_status)

                for callback in self.callbacks:
                    callback.on_loop_end()

                stop_condition_reached = self._stop_condition()

            # for FastBO, post-processing steps are required to finish full evaluation for
            # the top-n best-performing configurations, before printing the best metric found
            if isinstance(self.scheduler, FastBO) and \
                    self.scheduler.fastbo_manager.warmup_point != 1:
                self._fastbo_postprocess(running_trials_ids, done_trials_statuses)

        except Exception as e:
            logger.error(
                "An error happened during the tuning, cleaning up resources and logging final resources "
                "before throwing the exception."
            )
            raise e
        finally:
            # graceful termination block called when the tuner reached its stop condition, when an error happened or
            # when the job got interrupted (can happen in spot-instances or when sending a SIGINT signal with ctrl+C).
            # the block displays the best configuration found and stops trials that may still be running.
            # todo: here we output the best metric found from all the values during the whole process. Bue we also need
            #  the original version of the HPO algorithms.
            best_trialid, best_metric = print_best_metric_found(
                tuning_status=self.tuning_status,
                metric_names=self.scheduler.metric_names(),
                mode=self.scheduler.metric_mode(),
            )

            for event in self.trial_backend._simulator_state.event_heap:
                if isinstance(event[2], OnTrialResultEvent) and event[2].trial_id == best_trialid:
                    result = event[2].result
                    if 'val_accuracy' in result:
                        if result['val_accuracy'] > best_metric:
                            best_metric = result['val_accuracy']
                    elif 'metric_valid_error' in result:
                        if result['metric_valid_error'] < best_metric:
                            best_metric = result['metric_valid_error']
                    elif 'metric_valid_loss' in result:
                        if result['metric_valid_loss'] < best_metric:
                            best_metric = result['metric_valid_loss']

            print(f"find better metric {best_metric} for trial-id {best_trialid}")

            # Callbacks (typically includes writing final results)
            for callback in self.callbacks:
                callback.on_tuning_end()

            # Serialize Tuner object
            if self.save_tuner:
                self.save()

            logger.info("Stopping trials that may still be running.")
            self.trial_backend.stop_all()

            # notify tuning status that jobs were stopped without having to query their status in the backend since
            # we know that all trials were stopped
            self.tuning_status.mark_running_job_as_stopped()

            # in case too many errors were triggered, show log of last failed job and terminates with an error
            if self.tuning_status.num_trials_failed > self.max_failures:
                self._handle_failure(done_trials_statuses=done_trials_statuses)

            logger.info(
                f"Tuning finished, results of trials can be found on {self.tuner_path}"
            )

    def _sleep(self):
        time.sleep(self.sleep_time)
        for callback in self.callbacks:
            callback.on_tuning_sleep(self.sleep_time)

    @staticmethod
    def _set_metadata(metadata: Dict[str, Any], name: str, value):
        if name in metadata:
            logger.warning(
                f"Entry {name} in metadata is used, but will be overwritten:\n"
                f"Old value: {metadata[name]}\n"
                f"Overwrite: {value}\n"
            )
        metadata[name] = value

    def _enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param metadata: Original metadata
        :return: ``metadata`` enriched by default entries
        """
        res = metadata if metadata is not None else dict()
        self._set_metadata(res, ST_TUNER_CREATION_TIMESTAMP, time.time())
        self._set_metadata(res, "entrypoint", self.trial_backend.entrypoint_path().stem)
        self._set_metadata(res, "backend", str(type(self.trial_backend).__name__))
        for key, value in self.scheduler.metadata().items():
            self._set_metadata(res, key, value)
        return res

    def _save_metadata(self):
        dump_json_with_numpy(self.metadata, self.tuner_path / ST_METADATA_FILENAME)

    def _stop_condition(self) -> bool:
        return (
            self.stop_criterion(self.tuning_status)
            or self.tuning_status.num_trials_failed > self.max_failures
        )

    def _process_new_results(
        self, running_trials_ids: Set[int],
        is_final: Optional[bool] = False,
    ) -> (TrialAndStatusInformation, TrialIdAndResultList):
        """Communicates new results from the backend to the scheduler

        Returns dictionary of trials which are not running, along with their
        status, in ``done_trials_statuses``, and list of new results (tuples
        ``(trial_id, result)``), observed since the previous call, in
        ``new_results``.

        :param running_trials_ids: Trials currently running
        :return: ``(done_trials_statuses, new_results)``
        """

        # fetch new results
        trial_status_dict, new_results = self.trial_backend.fetch_status_results(
            trial_ids=list(running_trials_ids)
        )

        for callback in self.callbacks:
            callback.on_fetch_status_results(
                trial_status_dict=trial_status_dict, new_results=new_results
            )

        assert len(running_trials_ids) <= self.n_workers

        # Gets list of trials that are done with the new results.
        # The trials can be finished for different reasons:
        # - they completed,
        # - they were stopped independently of the scheduler, e.g. due to a
        #   timeout argument or a manual interruption
        # - scheduler decided to interrupt them.
        # Note: ``done_trials`` includes trials which are paused.
        done_trials_statuses = self._update_running_trials(
            trial_status_dict, new_results, is_final=is_final,
        )
        trial_status_dict.update(done_trials_statuses)

        # update status with new results and all done trials
        self.tuning_status.update(
            trial_status_dict=trial_status_dict, new_results=new_results
        )

        return done_trials_statuses, new_results

    def _schedule_new_tasks(self, running_trials_ids: Set[int]):
        """Schedules new tasks if resources are available or sleep.

        Note: If ``start_jobs_without_delay`` is False, we ask the backend for
        the number of busy workers, instead of trusting ``running_trials_ids``.
        The latter does not contain trials which have been stopped or completed,
        but the underlying job is still not completely done.

        :param running_trials_ids: set if trial-ids currently running, gets
            updated if new trials are scheduled.
        """
        running_trials_threshold = self.n_workers if self.asynchronous_scheduling else 1
        if self.start_jobs_without_delay:
            # Assume that only the trials in ``running_trial_ids`` are busy (which
            # is an underestimate for certain backends)
            busy_trial_ids = None
            num_busy_workers = len(running_trials_ids)
        else:
            # Ask backend how many workers are really busy
            busy_trial_ids = self.trial_backend.busy_trial_ids()
            num_busy_workers = len(busy_trial_ids)
        if num_busy_workers >= running_trials_threshold:
            # Note: For synchronous scheduling, we need to sleep here if at
            # least one worker is busy
            logger.debug(
                f"{num_busy_workers} of {self.n_workers} workers are "
                f"busy, wait for {self.sleep_time} seconds"
            )
            self._sleep()
        else:
            if not self.start_jobs_without_delay and num_busy_workers < len(
                running_trials_ids
            ):
                # In this case, the information from the backend is more recent
                running_trials_ids = set(x[0] for x in busy_trial_ids)
            # Schedule as many trials as we have free workers
            for _ in range(self.n_workers - num_busy_workers):
                trial = self._schedule_new_task()
                trial_id = trial.trial_id
                running_trials_ids.add(trial_id)
                # Update tuning status
                self.tuning_status.update(
                    trial_status_dict={trial_id: (trial, Status.in_progress)},
                    new_results=[],
                )

    def _schedule_new_task(self) -> Optional[TrialResult]:
        """Schedules a new task according to scheduler suggestion.

        :return: Information for the trial suggested, ``None`` if the scheduler does
            not suggest a new configuration (this can happen if its configuration
            space is exhausted)
        """
        suggestion = self.scheduler.suggest(trial_id=self.trial_backend.new_trial_id())
        if suggestion is None:
            logger.info("Searcher ran out of candidates, tuning job is stopping.")
            raise StopIteration
        elif suggestion.spawn_new_trial_id:
            # we schedule a new trial, possibly using the checkpoint of ``checkpoint_trial_id``
            # if given.
            trial = self.trial_backend.start_trial(
                config=suggestion.config.copy(),
                checkpoint_trial_id=suggestion.checkpoint_trial_id,
            )
            self.scheduler.on_trial_add(trial=trial) # it is no use from from my observation...
            logger.info(f"(trial {trial.trial_id}) - scheduled {suggestion}")
            return trial
        else:
            # suggestion is a trial_id to resume, with possibly a new configuration
            log_msg = f"Resuming trial {suggestion.checkpoint_trial_id}"
            if suggestion.config is not None:
                log_msg += f" with new_config = {suggestion.config}"
            logger.info(log_msg)
            trial = self.trial_backend.resume_trial(
                trial_id=suggestion.checkpoint_trial_id, new_config=suggestion.config
            )
            return trial

    def _schedule_final_tasks(self, best_trials_ids: Set[int],
                              running_trials_ids: Set[int]):
        """
        Schedule final tasks for the best-performing configurations. It is used only
        for FastBO's post-processing. This method is developed referred to
        `_schedule_new_tasks` and `_schedule_new_task`.
        """
        # Before scheduling final tasks for the `best_trials_ids`, we need to remove
        # the trials, whose pre-exponential point and saturation point are the same,
        # from `best_trials_ids`. Because in this case, we don't need to resume it.
        to_remove = set()
        for trial_id in best_trials_ids:
            if self.scheduler.searcher.trials_pre_exponential_points[str(trial_id)] == \
                    self.scheduler.searcher.trials_saturation_points[str(trial_id)]:
                to_remove.add(trial_id)
        best_trials_ids.difference_update(to_remove)

        assert (
            len(best_trials_ids) <= self.n_workers
        ), f"Selected trials to be resumed should be less than the number of workers, " \
           f"which is {self.n_workers}"

        # Schedule the trials
        for trial_id in best_trials_ids:
            suggestion = self.scheduler.suggest_final_resumed_task(trial_id=trial_id)
            if suggestion is None:
                logger.info("Searcher ran out of candidates, tuning job is stopping.")
                raise StopIteration
            else:
                # suggestion is a trial_id to resume, with possibly a new configuration
                log_msg = f"Resuming trial {suggestion.checkpoint_trial_id}"
                if suggestion.config is not None:
                    log_msg += f" with new_config = {suggestion.config}"
                logger.info(log_msg)
                trial = self.trial_backend.resume_trial(
                    trial_id=suggestion.checkpoint_trial_id, new_config=suggestion.config
                )

            running_trials_ids.add(trial_id)
            # Update tuning status
            self.tuning_status.update(
                trial_status_dict={trial_id: (trial, Status.in_progress)},
                new_results=[],
            )

    def _fastbo_postprocess(self, running_trials_ids, done_trials_statuses):
        """
        Post-process for FastBO: select the best-performing configurations
        and evaluate them to their saturation points. Contain 4 steps.
        """
        # post-processing step 1: pause the remaining trials
        if running_trials_ids is not None:
            new_done_trial_statuses = self._pause_running_trials(running_trials_ids)
            done_trials_statuses.update(new_done_trial_statuses)
            # fix: set its two points to default values
            self.scheduler.set_default_point_values(running_trials_ids)
            running_trials_ids.difference_update(new_done_trial_statuses.keys())

        # post-processing step 2: find the top-n best-performing trials that will be resumed
        best_trials_ids = find_best_n_trials_ids(
            tuning_status=self.tuning_status,
            metric_name=self.scheduler.metric_names()[0],
            n=4,
            mode=self.scheduler.metric_mode(),
        )

        # post-processing step 3: resume trials in `best_trials_ids`
        try:
            self._schedule_final_tasks(running_trials_ids=running_trials_ids,
                                       best_trials_ids=set(best_trials_ids)
                                       )
        except StopIteration:
            logger.info(
                "Tuning is finishing as the whole configuration space got exhausted."
            )
            # config_space_exhausted = True
            print(
                "Tuning is finishing as the whole configuration space got exhausted."
            )

        self.status_printer(self.tuning_status)

        # post-processing step 4: finish the trials, don't add new trials
        remaining_tasks_finished = bool(len(running_trials_ids) == 0)
        while not remaining_tasks_finished:
            new_done_trial_statuses, new_results = self._process_new_results(
                running_trials_ids=running_trials_ids, is_final=True,
            )

            if new_results and self.save_tuner:
                # Save tuner state only if there have been new results
                self.tuner_saver(tuner=self)

            done_trials_statuses.update(new_done_trial_statuses)
            running_trials_ids.difference_update(new_done_trial_statuses.keys())

            self._sleep()

            self.status_printer(self.tuning_status)

            remaining_tasks_finished = bool(len(running_trials_ids) == 0)

    def _handle_failure(self, done_trials_statuses: Dict[int, Tuple[Trial, str]]):
        logger.error(f"Stopped as {self.max_failures} failures were reached")
        for trial_id, (_, status) in done_trials_statuses.items():
            if status == Status.failed:
                logger.error(f"showing log of first failure")
                stdout = "".join(self.trial_backend.stdout(trial_id))
                stderr = "".join(self.trial_backend.stderr(trial_id))
                logger.error(stdout)
                logger.error(stderr)
                raise ValueError(f"Trial - {trial_id} failed")

    def save(self, folder: Optional[str] = None):
        if folder is None:
            tuner_serialized_path = self.tuner_path / ST_TUNER_DILL_FILENAME
        else:
            tuner_serialized_path = Path(folder) / ST_TUNER_DILL_FILENAME
        with open(tuner_serialized_path, "wb") as f:
            logger.debug(f"saving tuner in {tuner_serialized_path}")
            dill.dump(self, f)
            self.trial_backend.on_tuner_save()  # callback

    @staticmethod
    def load(tuner_path: Optional[str]):
        with open(Path(tuner_path) / ST_TUNER_DILL_FILENAME, "rb") as f:
            tuner = dill.load(f)
            tuner.tuner_path = Path(experiment_path(tuner_name=tuner.name))
            return tuner

    def _update_running_trials(
        self,
        trial_status_dict: TrialAndStatusInformation,
        new_results: TrialIdAndResultList,
        is_final: Optional[bool] = False,
    ) -> TrialAndStatusInformation:
        """
        Updates schedulers with new results and sends decision to stop/pause
        trials to the backend. Trials can be finished because:

        * the scheduler decided to stop or pause.
        * the trial failed.
        * the trial was stopped independently of the scheduler, e.g. due to a
          timeout argument or a manual interruption.
        * the trial completed.

        :param trial_status_dict: Information on trials from
            ``trial_backend.fetch_status_results``
        :param new_results: New results from ``trial_backend.fetch_status_results``
        :param callbacks: ``on_trial_result`` for these callbacks is called here
        :return: Dictionary mapping trial-ids that are finished to status
        """
        # gets the list of jobs from running_jobs that are done
        done_trials = dict()

        for trial_id, result in new_results:
            if trial_id not in done_trials:
                trial, status = trial_status_dict[trial_id]

                # communicate new result to the searcher and the scheduler
                self.last_seen_result_per_trial[trial_id] = result
                if is_final and \
                        isinstance(self.scheduler, FastBO) and \
                        self.scheduler.fastbo_manager.warmup_point != 1:
                    decision = self.scheduler.final_on_trial_result(trial=trial, result=result)
                else:
                    decision = self.scheduler.on_trial_result(trial=trial, result=result)

                for callback in self.callbacks:
                    callback.on_trial_result(
                        trial=trial,
                        status=status,
                        result=result,
                        decision=decision,
                    )

                if decision == SchedulerDecision.STOP:
                    if status != Status.completed:
                        # we override the status immediately, this avoids calling the backend status another time to
                        # update after the change which may be expensive
                        status = Status.stopped
                        self.trial_backend.stop_trial(trial_id=trial_id, result=result)
                    self.scheduler.on_trial_remove(trial=trial)
                    done_trials[trial_id] = (trial, status)
                    self.trials_scheduler_stopped.add(trial_id)

                elif decision == SchedulerDecision.PAUSE:
                    status = Status.paused
                    self.trial_backend.pause_trial(trial_id=trial_id, result=result)
                    self.scheduler.on_trial_remove(trial=trial)
                    done_trials[trial_id] = (trial, status)

        for trial_id, (trial, status) in trial_status_dict.items():
            # Status "completed", "stopped" and "failed" are signaled to scheduler.
            # Status "in_progress" and "stopping" are not signaled, although the first one could be added
            # to notify the scheduler of pending runtimes (even in the absence of new results).

            if status == Status.completed:
                # since the code above updates ``trial_status_dict[trial_id]`` after a pause/stop scheduling decision
                # this callback is never called after a pause/stop scheduler decision.
                if (
                    trial_id not in done_trials
                    or done_trials[trial_id][1] != Status.paused
                ):
                    logger.info(f"Trial trial_id {trial_id} completed.")
                # If scheduler marks trial as ``Status.paused``, this overrides
                # ``Status.completed`` (which was assigned because the job
                # completed)
                done_trial = done_trials.get(trial_id)
                if done_trial is not None and done_trial[1] == Status.paused:
                    status = Status.paused
                if trial_id not in self.last_seen_result_per_trial:
                    logger.error(
                        f"trial {trial_id} completed and no metrics got observed, corresponding log:"
                    )
                    stdout = "".join(self.trial_backend.stdout(trial_id))
                    stderr = "".join(self.trial_backend.stderr(trial_id))
                    logger.error(stdout)
                    logger.error(stderr)
                    raise ValueError(
                        f"trial {trial_id} completed and no metrics got observed"
                    )

                last_result = self.last_seen_result_per_trial[trial_id]
                if trial_id not in done_trials:
                    self.scheduler.on_trial_complete(trial, last_result)
                if status == Status.completed:
                    for callback in self.callbacks:
                        callback.on_trial_complete(trial, last_result)
                done_trials[trial_id] = (trial, status)

            if status == Status.failed:
                logger.info(f"Trial trial_id {trial_id} failed.")
                self.scheduler.on_trial_error(trial)
                done_trials[trial_id] = (trial, status)

            # For the case when the trial is stopped independently of the scheduler, we choose to use
            # scheduler.on_trial_error(...) since it was not the scheduler's decision to stop the trial.
            if (
                status == Status.stopped
                and trial_id not in self.trials_scheduler_stopped
            ):
                logger.info(
                    f"Trial trial_id {trial_id} was stopped independently of the scheduler."
                )
                self.scheduler.on_trial_error(trial)
                done_trials[trial_id] = (trial, status)

        return done_trials

    def _pause_running_trials(
        self,
        running_trials_ids: Set[int],
    ) -> TrialAndStatusInformation:
        """
        Pauses the remaining running trials after the main while loop in `run` method.
        This is only used in FastBO, to finish full evaluation for some well-performing trials.
        Don't need to ask the scheduler for the decision (i.e., always pause)
        Note: this method is actually like `_process_new_results`, but uses a simplified and
        modified `_update_running_trials`. See the comment below.
        """
        trial_status_dict, new_results = self.trial_backend.fetch_last_trial_information(
            trial_ids=list(running_trials_ids)
        )

        # The following part is like `_update_running_trials` in Tuner, just keeping the related
        # part for decision == pause.
        # 1. Sometimes the size of `trial_status_dict` (or the size of `running_trials_ids`)
        # can be larger than the size of `new_results`. That is because a trial may have been
        # started but its first result hasn't been added in to `self._trial_dict` due to the stop
        # condition.
        # 2. Using `trial_backend.pause_trial` may lead to unexpected runs of subsequent epochs,
        # because trials in `trial_status_dict` are processed one by one. To address this issue,
        # I designed and employed a new method `pause_all_trials`.
        # 3. For "direct" mode that directly uses specified pre-exponential and saturation points,
        # we need to update `trials_pre_exponential_points` and `trials_saturation_points` for the
        # trials to be paused.
        done_trials = dict()
        for trial_id in trial_status_dict:
            trial, status = trial_status_dict[trial_id]

            status = Status.paused
            if self.scheduler.fastbo_manager.mode == "direct":
                self.scheduler.searcher.trials_pre_exponential_points[str(trial_id)] = \
                    int(self.scheduler._max_resource_level * self.scheduler.fastbo_manager.warmup_point)
                self.scheduler.searcher.trials_saturation_points[str(trial_id)] = \
                    int(self.scheduler._max_resource_level * self.scheduler.fastbo_manager.saturation_point)
            # self.trial_backend.pause_trial(trial_id=trial_id)
            self.scheduler.on_trial_remove(trial=trial)  # none, defined in `TrialScheduler`
            done_trials[trial_id] = (trial, status)
        self.trial_backend.pause_all_trials(trial_ids=running_trials_ids, results=new_results)
        trial_status_dict.update(done_trials)

        # After `_updating_running_trials`, we don't need to update status with new results and all
        # done trials as in `_process_new_results`. Because we just get `trial_status_dict` and
        # `new_results` from `trial_backend._trial_dict`, rather than some newly generated results.
        return done_trials

    @staticmethod
    def _default_callback():
        """
        :return: Default callback to store results
        """
        return StoreResultsCallback()
