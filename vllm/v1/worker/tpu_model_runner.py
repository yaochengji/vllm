import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast
from unittest.mock import patch

import torch
import torch.distributed
import torch.nn as nn
# TorchAx related
import torchax
import jax

from vllm.attention import AttentionMetadata, Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasAttentionBackendImpl,
                                               PallasMetadata)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.model_runner_base import ExecutionMode, ModelRunnerBase

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
# FIXME(woosuk): Temporarily disabled top-p sampling since it's too slow.
_ENABLE_TOP_P = False
# FIXME(woosuk): A temporary hack to support `n > 1`.
# This can significantly affect the performance if too large.
_MAX_NUM_SAMPLES = 128

def get_all_submodules(module: nn.Module, prefix: str = "", memo=None):
    """
    Recursively iterates through a PyTorch nn.Module and yields all its submodules.

    Args:
        module: The PyTorch module to iterate through.
        prefix: (str, optional) A prefix for the module name (used in recursion).  Defaults to "".
        memo: (set, optional) A set to keep track of visited modules to handle shared modules correctly.
    """
    if memo is None:
        memo = set()

    if id(module) in memo:
        return
    memo.add(id(module))

    for name, child in module.named_children():
        full_name = f"{prefix}{name}" if prefix else name
        yield full_name, child  # Yield the current child
        yield from get_all_submodules(
            child, full_name + ".", memo
        )  # Recursively call on the child


@dataclass
class PromptInputData:

    req_ids: List
    prompt_lens: List
    input_tokens: List
    input_positions: List
    attn_metadata: List

    def zipped(self):
        return zip(self.req_ids, self.prompt_lens, self.input_tokens,
                   self.input_positions, self.attn_metadata)


@dataclass
class DecodeInputData:
    req_ids: List
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[PallasMetadata] = None


class TPUModelRunner(ModelRunnerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Used to initialize positions for the individual prefills
        self.prefill_input_positions = torch.tensor(range(self.max_model_len),
                                                    device="cpu",
                                                    dtype=torch.int32).reshape(
                                                        1, -1)

    def _prepare_prompt_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptInputData:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        req_ids = []
        prompt_lens = []
        input_tokens_list = []
        input_positions_list = []
        attn_metadata_list = []
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            num_computed_tokens = req_state.num_computed_tokens
            num_prompt_tokens = len(req_state.prompt_token_ids)

            # Detect whether this is a prompt (can be full or chunked)
            if num_computed_tokens >= num_prompt_tokens:
                # This is a decode => Skip
                continue

            # This is a prompt
            req_ids.append(req_id)

            # Prompt len
            prompt_len = num_scheduled_tokens
            prompt_lens.append(prompt_len)
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            assert padded_prompt_len <= self.max_model_len

            # Seq len
            seq_len = num_computed_tokens + prompt_len

            # Input tokens
            input_tokens = torch.zeros((1, padded_prompt_len),
                                       dtype=torch.int32,
                                       device="cpu")
            input_tokens[:, :prompt_len] = torch.from_numpy(
                self.input_batch.token_ids_cpu[req_index,
                                               num_computed_tokens:seq_len])
            # input_tokens = torch.from_numpy(self.input_batch.token_ids_cpu[
            #     req_index, num_computed_tokens:padded_seq_len].reshape(1, -1))
            # input_tokens[:, prompt_len:] = 0
            input_tokens_list.append(input_tokens.to(self.device))

            # Input positions
            input_positions = torch.zeros((1, padded_prompt_len),
                                          dtype=torch.int32,
                                          device="cpu")
            input_positions[:, :
                            prompt_len] = self.prefill_input_positions[:,
                                                                       num_computed_tokens:
                                                                       seq_len]
            # input_positions[:, prompt_len:] = 0
            input_positions_list.append(input_positions.to(self.device))

            # Slot mapping
            block_table_cpu_tensor = \
                self.input_batch.block_table.get_cpu_tensor()
            block_numbers = block_table_cpu_tensor[req_index,
                                                   input_positions //
                                                   self.block_size].reshape(
                                                       1, -1)

            block_offsets = input_positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            slot_mapping[:, prompt_len:] = _PAD_SLOT_ID
            slot_mapping = slot_mapping.long()

            # Block table
            block_table = None
            if num_computed_tokens > 0:
                block_table = block_table_cpu_tensor[req_index].unsqueeze(0)
                block_table = block_table.to(self.device)

            # Context len
            context_len = 0
            if num_computed_tokens > 0:
                context_len = seq_len
            context_lens = torch.tensor([context_len],
                                        dtype=torch.int32,
                                        device="cpu")

            # Effective query len
            effective_query_lens = torch.tensor([prompt_len],
                                                dtype=torch.int32,
                                                device="cpu")

            # Attn metadata
            attn_metadata_list.append(
                PallasMetadata(
                    num_prefills=1,
                    num_prefill_tokens=0,  # NOTE: This is not used.
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping.to(self.device),
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=block_table,
                    context_lens=context_lens.to(self.device),
                    effective_query_lens=effective_query_lens.to(self.device),
                ))

            # TODO: Remove this
            # if num_computed_tokens > 0:
            #     print("-------------------")
            #     print("input_tokens.shape = {}".format(input_tokens.shape))
            #     print("input_positions.shape = {}".format(
            #         input_positions.shape))
            #     print("slot_mapping.shape = {}".format(slot_mapping.shape))
            #     print("block_table.shape = {}".format(block_table.shape))
            #     print("context_lens.shape = {} data = {}".format(
            #         context_lens.shape, context_lens))
            #     print("effective_query_lens.shape = {} data = {}".format(
            #         effective_query_lens.shape, effective_query_lens))

        return PromptInputData(
            req_ids=req_ids,
            prompt_lens=prompt_lens,
            input_tokens=input_tokens_list,
            input_positions=input_positions_list,
            attn_metadata=attn_metadata_list,
        )

    def _prepare_decode_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> DecodeInputData:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        block_table_cpu_tensor = self.input_batch.block_table.get_cpu_tensor()

        req_ids = []
        req_indices = []
        input_tokens = []
        input_positions = []
        slot_mapping = []
        context_lens = []
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            num_computed_tokens = req_state.num_computed_tokens
            num_prompt_tokens = len(req_state.prompt_token_ids)

            # Detect whether this is a decode
            if num_computed_tokens < num_prompt_tokens:
                # This is a prompt => Skip
                continue

            # This is a decode
            req_ids.append(req_id)
            req_indices.append(req_index)

            # Seq len
            seq_len = num_computed_tokens + num_scheduled_tokens

            # Sanity check decode
            assert num_scheduled_tokens == 1
            assert seq_len == req_state.num_tokens

            # Input token
            input_tokens.append([
                self.input_batch.token_ids_cpu[req_index, num_computed_tokens]
            ])

            # Position
            input_positions.append([num_computed_tokens])

            # Slot mapping
            block_number = block_table_cpu_tensor[req_index,
                                                  num_computed_tokens //
                                                  self.block_size]
            block_offset = num_computed_tokens % self.block_size
            slot_id = block_number * self.block_size + block_offset
            slot_mapping.append([slot_id])

            # Context len
            context_lens.append(seq_len)

        # Compute padding
        batch_size = len(input_tokens)
        padded_batch_size = _get_padded_batch_size(batch_size)
        num_padding = padded_batch_size - batch_size

        # Add padding
        input_tokens.extend([[0]] * num_padding)
        input_positions.extend([[0]] * num_padding)
        slot_mapping.extend([[_PAD_SLOT_ID]] * num_padding)
        context_lens.extend([0] * num_padding)
        req_indices.extend([0] * num_padding)

        # Create tensors
        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.int32,
                                           device="cpu")
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.int32,
                                              device="cpu")
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.int64,
                                           device="cpu")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int32,
                                           device="cpu")
        block_tables_tensor = block_table_cpu_tensor[req_indices]

        # Attn metadata
        attn_metadata = PallasMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=padded_batch_size,
            slot_mapping=slot_mapping_tensor.to(self.device),
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            block_tables=block_tables_tensor.to(self.device),
            context_lens=context_lens_tensor.to(self.device),
            effective_query_lens=None,
        )

        return DecodeInputData(
            req_ids=req_ids,
            input_tokens=input_tokens_tensor.to(self.device),
            input_positions=input_positions_tensor.to(self.device),
            attn_metadata=attn_metadata)

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # Update cached state
        self.update_states(scheduler_output)

        # Prepare inputs
        prompt_data = self._prepare_prompt_inputs(scheduler_output)
        decode_data = self._prepare_decode_inputs(scheduler_output)

        # Init
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        sampled_token_ids_list = [0] * num_reqs

        # Run decodes (a single batch)
        if len(decode_data.req_ids) > 0:
            # Forward
            print("decode_data.attn_metadata.slot_mapping: ", decode_data.attn_metadata.slot_mapping)
            with set_forward_context(decode_data.attn_metadata,
                                     self.vllm_config):
                assert self.model is not None
                selected_token_ids, updated_kv_caches = self.model(decode_data.input_tokens,
                                                decode_data.input_positions,
                                                decode_data.attn_metadata,
                                                self.kv_caches)
                # print("decoding updated_kv_caches: ", updated_kv_caches[0][0].to(torch.float32).sum())
                self.kv_caches = updated_kv_caches
                print("decoding updated_kv_caches: ", self.kv_caches[0][0][0][0][:, 0])
                for cache_idx, cache_name in enumerate(self.vllm_config.compilation_config.static_forward_context.keys()):
                    self.vllm_config.compilation_config.static_forward_context[cache_name].kv_cache = [updated_kv_caches[cache_idx]]

            # Transfer sampled tokens from TPU to CPU
            selected_token_ids_list = selected_token_ids.cpu().tolist()

            # Update cached state
            for i, req_id in enumerate(decode_data.req_ids):
                req_index = self.input_batch.req_id_to_index[req_id]
                req_state = self.requests[req_id]

                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])

                token_id = selected_token_ids_list[i]

                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

                sampled_token_ids_list[req_index] = token_id

        # Run each prompt
        for (req_id, prompt_len, input_tokens, input_positions,
             attn_metadata) in prompt_data.zipped():
            assert req_id is not None
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Forward
            with set_forward_context(attn_metadata, self.vllm_config):
                assert self.model is not None
                selected_token_ids, updated_kv_caches = self.model(input_tokens, input_positions,
                                                attn_metadata, self.kv_caches)
                self.kv_caches = updated_kv_caches
                print("prompt updated_kv_caches: ", self.kv_caches[0][0][0][0][:, 0])
                for cache_idx, cache_name in enumerate(self.vllm_config.compilation_config.static_forward_context.keys()):
                    self.vllm_config.compilation_config.static_forward_context[cache_name].kv_cache = [updated_kv_caches[cache_idx]]

            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= len(req_state.prompt_token_ids):
                # Transfer sampled tokens from TPU to CPU
                token_id = selected_token_ids.cpu()[prompt_len - 1].item()
                sampled_token_ids_list[req_index] = token_id

                # Update cached state
                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

        # Get req_ids
        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(List[str], self.input_batch.req_ids[:num_reqs])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids_list,
            logprob_token_ids_cpu=None,
            logprobs_cpu=None,
        )

        return model_runner_output

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.
        # TODO(chengjiyao): verify this could work for both SPMD and mannual collective
        tp_rank = jax.process_index()
        with patch(
                "vllm.model_executor.layers.vocab_parallel_embedding."
                "get_tensor_model_parallel_rank",
                return_value=tp_rank):
            model = get_model(vllm_config=self.vllm_config)
        model = model.eval()
        self.model_wrapper = ModelWrapperV1(model, self)
        # self.model = self.model_wrapper
        self.model = torchax.compile(self.model_wrapper)

    def dummy_run(
        self,
        kv_caches,
        num_tokens: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
        sync=False,
    ) -> None:
        assert seq_len is not None
        assert exec_mode is not None

        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )

            else:
                context_lens = torch.ones((num_tokens, ),
                                          dtype=torch.int32,
                                          device=self.device)

                block_tables = torch.zeros(
                    (num_tokens, self.max_num_blocks_per_req),
                    dtype=torch.int32,
                    device=self.device)

                effective_query_lens = torch.ones_like(context_lens)

                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
        else:
            assert seq_len == 1
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (num_tokens, self.max_num_blocks_per_req),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((num_tokens, ),
                                      dtype=torch.int32,
                                      device=self.device)
            attn_metadata = PallasMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_tokens * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=True,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        # TODO: Remove the attn_metadata above
        with set_forward_context(None, self.vllm_config):
            assert self.model is not None
            model_result = self.model(token_ids, position_ids, None, kv_caches)
            if sync:
                torchax.interop.call_jax(jax.block_until_ready, model_result)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        if len(kv_cache_config.groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: Dict[str, torch.Tensor] = {}

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype

                tpu_k_cache = torch.zeros(kv_cache_shape,
                                          dtype=dtype,
                                          device=self.device)
                tpu_v_cache = torch.zeros_like(tpu_k_cache)

                kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)


class ModelWrapperV1(nn.Module):

    def __init__(self, model: nn.Module, runner: TPUModelRunner):
        super().__init__()
        self.model = model
        self.runner = runner

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The Pallas attention metadata.
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
            num_samples: Number of samples to draw from each logits vector.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        # Skip this in memory profiling at initialization.
        # index_copy_(slot_mapping) only works when the inserted dimension
        # is 0. However, the KV cache in the Pallas backend has the shape
        # [num_kv_heads, num_blocks, block_size, head_size]. To make it
        # work, we need to flatten the first three dimensions and modify
        # the slot_mapping accordingly.
        num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
        slot_mapping = attn_metadata.slot_mapping
        slot_mapping = slot_mapping.flatten()
        head_indicies = torch.arange(0,
                                        num_kv_heads,
                                        device=slot_mapping.device,
                                        dtype=slot_mapping.dtype)
        head_indicies *= block_size * num_blocks
        slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
            -1, num_kv_heads)
        slot_mapping = slot_mapping + head_indicies.view(1, -1)
        slot_mapping = slot_mapping.flatten()
        attn_metadata.slot_mapping = slot_mapping
        print("attn_metadata.slot_mapping.shape: ", attn_metadata.slot_mapping.shape)

        assert self.model is not None
        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )

        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        updated_kv_cache = self._get_updated_kv_cache()
        updated_kv_cache = kv_caches if updated_kv_cache is None else updated_kv_cache
        return argmax_token_ids, updated_kv_cache

    def _get_updated_kv_cache(self):
        updated_kv_caches = []
        for _, submodule in get_all_submodules(self.model):
            if isinstance(submodule, Attention):
                if submodule.impl.updated_kv_cache is not None:
                    updated_kv_caches.append(submodule.impl.updated_kv_cache)
                else:
                    return
        return updated_kv_caches
    

def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16
