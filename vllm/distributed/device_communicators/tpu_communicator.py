# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Sequence

import torch
from torch.distributed import ProcessGroup

from vllm.platforms import current_platform

from .base_device_communicator import DeviceCommunicatorBase

if current_platform.is_tpu():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla._internal import pjrt
    from torch_xla.distributed.xla_multiprocessing import create_optimized_replica_groups

    from vllm.executor import ray_utils


@torch.library.custom_op("tpu::all_gather", mutates_args=())
def tpu_all_gather(input_: torch.Tensor, dim: int, groups: Sequence[int]) -> torch.Tensor:
    return xm.all_gather(input_, dim=dim, groups=[groups], pin_layout=False,
                             channel_id=1, use_global_device_ids=True)

@tpu_all_gather.register_fake
def _(input_: torch.Tensor, dim: int, groups: Sequence[int]):
    input_shape = list(input_.shape)
    output_shape = input_shape[:]
    output_shape[dim] = input_shape[dim] * len(groups)
    return input_.new_empty(*output_shape)


@torch.library.custom_op("tpu::reduce_scatter", mutates_args=())
def tpu_reduce_scatter(input_: torch.Tensor, groups: Sequence[int]) -> torch.Tensor:
    return xm.reduce_scatter(xm.REDUCE_SUM, input_, scale=1.0, scatter_dim=0, 
                             shard_count=len(groups), groups=[groups],
                             pin_layout=False, channel_id=1, use_global_device_ids=True)

@tpu_reduce_scatter.register_fake
def _(input_: torch.Tensor, groups: Sequence[int]):
    input_shape = list(input_.shape)
    assert input_shape[0] % len(groups) == 0
    output_shape = input_shape[:]
    output_shape[0] = input_shape[0] // len(groups)
    return input_.new_empty(*output_shape)


class TpuCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)

        # NOTE(woosuk): When using TP > 1 on TPUs, every TPU on the same node
        # must be used together. Therefore, the local rank and world size can
        # be simply calculated as follows.
        global_rank = self.global_rank
        global_world_size = self.global_world_size

        # Calculate how many TPU nodes are in the current deployment. This
        # is the Ray placement group if it is deployed with Ray. Default
        # to the number of TPU nodes in the Ray cluster. The number of TPU
        # nodes is computed by the total number of TPUs divided by the
        # number of TPU accelerators per node, to account for clusters
        # with both CPUs and TPUs.
        num_nodes = ray_utils.get_num_tpu_nodes()
        num_nodes_in_pg = ray_utils.get_num_nodes_in_placement_group()
        if num_nodes_in_pg > 0:
            num_nodes = num_nodes_in_pg

        local_world_size = global_world_size // num_nodes
        local_rank = global_rank % local_world_size

        # Ensure environment variables are set for multihost deployments.
        # On GKE, this is needed for libtpu and TPU driver to know which TPU
        # chip is actually visible. Otherwise the TPU driver will fail to
        # initialize because the number of devices would be different from
        # the number of visible worker addresses.
        os.environ["CLOUD_TPU_TASK_ID"] = str(global_rank)
        os.environ["TPU_VISIBLE_CHIPS"] = str(local_rank)

        pjrt.initialize_multiprocess(local_rank, local_world_size)
        xr._init_world_size_ordinal()

        self.optimized_replica_groups = create_optimized_replica_groups()
        if self.optimized_replica_groups is None:
            self.optimized_replica_groups = [[i for i in range(global_world_size)]]
        self.optimized_replica_groups = self.optimized_replica_groups[0]

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return xm.all_reduce(xm.REDUCE_SUM, input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return tpu_all_gather(input_, dim, self.optimized_replica_groups)

    def reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:
        return tpu_reduce_scatter(input_, self.optimized_replica_groups)
