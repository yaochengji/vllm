"""A TPU worker class."""
import os
from typing import Optional

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.tpu_model_runner import ExecutionMode, TPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

import torchax
env = torchax.enable_globally()
# env.config.debug_print_each_op = True

logger = init_logger(__name__)


class TPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method)
        # TOOD(chengjiyao): clean the monkey patch
        torch._sync = lambda *args, **kwargs: None

    def init_device(self):
        os.environ["PJRT_DEVICE"] = "TPU"
        torch.set_grad_enabled(False)
        torch.set_default_dtype(self.model_config.dtype)

        # Initialize the distributed environment.
        init_tpu_worker_distributed_environment(self.parallel_config,
                                                self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        # Device initialization should happen after initializing
        # the distributed runtime.
        self.device = torch.device("jax:0")
        self.device_config.device = self.device

        # Set random seed.
        set_random_seed(self.model_config.seed)
        # xm.set_rng_state(self.model_config.seed, self.device)

        # TODO(chengjiyao): set jax cache size limit
        # Use persistent cache to avoid XLA recompilation.
        # TODO(chengjiyao): do we need to set different cache for different rank

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner = TPUModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        assert self.model_runner is not None

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        kv_caches = [(torch.tensor([], dtype=torch.float32,
                                   device=self.device),
                      torch.tensor([], dtype=torch.float32,
                                   device=self.device))
                     for _ in range(num_layers)]

        # self.model_runner.dummy_run(
        #     kv_caches,
        #     num_tokens=1,
        #     seq_len=self.scheduler_config.max_num_batched_tokens,
        #     exec_mode=ExecutionMode.PREFILL,
        #     sync=True,
        # )
        

        # Get the maximum amount of memory used by the model weights and
        # intermediate activations.
        # TODO(chengjiyao): get memory info in jax
        total_memory_size = 32 * 1024 ** 3 # 32GB
        profiled = 24 * 1024 ** 3 # 24GB

        # Calculate the TPU KV cache size based on profiling.
        usable_memory_size = int(total_memory_size *
                                 self.cache_config.gpu_memory_utilization)
        tpu_kv_cache_bytes = max(usable_memory_size - profiled, 0)

        return int(tpu_kv_cache_bytes)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        assert self.model_runner is not None
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None


def init_tpu_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""

    # NOTE(woosuk): This is just to initialize the TP group and broadcast
    # the input objects on CPU. The all-reduce and all-gather ops on TPU
    # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
    # own context.
    # TODO(chengjiyao): should we change the backend to jax?
    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method=distributed_init_method,
        backend="gloo",
    )
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
