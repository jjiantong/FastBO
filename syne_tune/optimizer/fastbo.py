# Copyright 2023 Jiantong Jiang, The University of Western Australia
# Licensed under the Apache License, Version 2.0 (the "License").
#
# This file is created by Jiantong Jiang to implement FastBO
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
from typing import Dict, Any
import logging
from syne_tune.optimizer.schedulers.fastbo import FastBOScheduler

logger = logging.getLogger(__name__)


def _assert_searcher_must_be(kwargs: Dict[str, Any], name: str):
    searcher = kwargs.get("searcher")
    assert searcher is None or searcher == name, f"Must have searcher='{name}'"


class FastBO(FastBOScheduler):
    """Gaussian process based Bayesian optimization, partial evaluation.

    :param config_space: Configuration space for evaluation function
    :param metric: Name of metric to optimize
    :param kwargs: Additional arguments to
        :class:`~syne_tune.optimizer.schedulers.FIFOScheduler`
    """

    def __init__(self, config_space: Dict[str, Any], metric: str, **kwargs):
        searcher_name = "fastbo"
        _assert_searcher_must_be(kwargs, searcher_name)
        super(FastBO, self).__init__(
            config_space=config_space,
            metric=metric,
            searcher=searcher_name,
            **kwargs,
        )