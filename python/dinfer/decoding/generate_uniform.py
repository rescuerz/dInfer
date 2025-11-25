import torch
import numpy as np
import logging
import random

from transformers.models.layoutlmv2.modeling_layoutlmv2 import relative_position_bucket

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num, add_gumbel_noise_power, get_num_transfer_tokens

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

def select_undecoded(seq_idx, orig_x, x, block, block_loc, mask_id, writeback=False):
    """ 选择未完成解码的序列。

    在批量解码过程中，某些序列可能比其他序列先完成解码（即不再包含mask token）。
    此函数用于筛选出仍需解码的序列，并可选地将已完成解码的序列写回原始token数组。
    """
    if x.batch_size == 1:
        return seq_idx, x
    bool_idx = torch.all(block != mask_id, dim=1)

    if writeback:
        # Write the decoded tokens back
        finished_idx = seq_idx[bool_idx]
        orig_x[finished_idx, block_loc.start:block_loc.end] = block[bool_idx]

    # Select the undecoded sequences
    return seq_idx, x

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        # 初始筛选未解码序列
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        while (block == decoder.mask_id).sum() > 0:
            # 限制forward的次数，计算最大的forward次数
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

            # If there are more than one sequence, we should filter the sequences and only decode
            # on the sequences that still have masked tokens.
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class BlockDiffusionRunner(BlockRunner):
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : BlockDiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf, backend):
        super().__init__(diff_iteration, early_stop, maximum_unroll, expected_tpf)
        self.backend = backend

    def prefill(self, model, block, kv_cache, pos_ids, attn_mask):
        """ Prefill for KV Cache
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        block : torch.Tensor
            The input IDs of the tokens in the prefilling range.
        kv_cache: KVCache
            The KV-cache
        pos_ids: torch.Tensor
            The position IDs of the tokens in the prefilling range.
        attn_mask: torch.Tensor
            The attention mask of the tokens in the prefilling range.
        """
        if kv_cache is None:
            return
        else:
            # 执行一次前向传播以获取 KV cache
            output = model(block.clone(memory_format=torch.contiguous_format), use_cache=True, attention_mask=attn_mask, position_ids=pos_ids.clone(memory_format=torch.contiguous_format))
            if self.backend == 'vllm':
                kv_cache.update(output.past_key_values)
            else:
                kv_cache.range_update(output.past_key_values, 0, block.size(1), 0)
            self.diff_iteration.num_forwards +=1
            self.diff_iteration.iter_no +=1

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size

        # 准备 KV cache
        if kv_cache is not None:
            kv_cache.extend_cache(block_loc.end)
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
        else:
            past_key_values, replace_position = None, None

        input_block_mask_number = 0
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 2))
            for unroll_i in range(unroll_k):
                input_block_mask_number = (block == decoder.mask_id).sum()
                output = self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self.backend)
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
        # additional forward to update kvcache for the last decoding step in the current block
        # 额外的一次前向传播，用于更新当前块最后一步解码的 KV cache
        if kv_cache is not None:
            if input_block_mask_number > 0:
                output = model(block.clone(memory_format=torch.contiguous_format), 
                    past_key_values=past_key_values,
                    use_cache=True, 
                    position_ids=pos_ids[:, block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                    replace_position=(0,0) if self.backend=='sglang' else replace_position)
                self.diff_iteration.num_forwards +=1
                self.diff_iteration.iter_no +=1
            if self.backend=='vllm':
                kv_cache.update(output.past_key_values)
            else:
                kv_cache.range_update(output.past_key_values, 0, block_loc.end, block_loc.end - block_loc.start)



        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        cache_update_kv = None
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            cache_update_kv = output.past_key_values
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        # 根据 KV cache 的状态和类型执行不同的前向传播逻辑
        if kv_cache is None:
            # 如果没有 KV cache，直接对整个输入进行前向传播
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            # 如果是前缀缓存，获取对应的 KV cache 和替换位置
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # 仅对当前块及其后续部分进行前向传播
            logits = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            # 其他类型的缓存（如双向缓存）
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return cache_update_kv, logits

class BlockDiffusionIteration:
    """ An implementation of block diffusion iteration to decode.
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        past_key_values: List[List[torch.Tensor]]
            The key-values required to decode the specified block.
        replace_position: torch.Tensor 
            The tensor indicates the valid locations in the returned key-values.
        """
        if kv_cache is None:
            output = model(x.data[:, :block_loc.end], 
                attention_mask=attn_mask[:,:block_loc.end,:block_loc.end],
                position_ids=pos_ids[:, :block_loc.end])
            logits = output.logits[:, block_loc.start:block_loc.end]
        else:
            output = model(block.clone(memory_format=torch.contiguous_format),
                position_ids=pos_ids[:,block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                use_cache=True,
                past_key_values=past_key_values,
                replace_position=(0,0) if backend=='sglang' else replace_position)
            logits = output.logits
            # TODO(dulun): we don't need update kv cache for every step.
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
            
        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return output


class ShiftDiffusionIteration(DiffusionIteration):
    """ A shift implementation of diffusion iteration to decode.
    """
    def __init__(self, use_shift = False):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        # 计算移位后的块起始和结束位置
        block_start, block_end = block_loc.start-1, block_loc.end-1
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_start, block_end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            # TODO(dulun): need to improve efficiency
            # 创建移位后的 TokenArray
            x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
            # 使用生成的 logits 解码移位后的 TokenArray
            decoder.decode(output.logits[:, block_start:block_end], block_start, block_end, x_shifted)
            # 将解码结果写回原始 TokenArray
            x.data[:, 1:] = x_shifted.data
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        # 根据 KV cache 的状态和类型执行不同的前向传播逻辑
        if kv_cache is None:
            logits = model(x.data).logits[:, block_start:block_end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_end - block_start
            logits = logits[:, :block_length]
        else:
            # cache position is the position between current_block_start and current_block_end
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:block_end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        # TODO(dulun): need to improve efficiency
        # 再次创建移位后的 TokenArray 并解码
        x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
        decoder.decode(logits, block_start, block_end, x_shifted)
        x.data[:, 1:] = x_shifted.data
        self.num_forwards += 1
        self.iter_no += 1

class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, use_shift=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        # 根据是否使用移位选择不同的扩散迭代实现
        if use_shift:
            self.diff_iteration = ShiftDiffusionIteration()
        else:
            self.diff_iteration = BaseDiffusionIteration()
        # 初始化块解码器
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # 初始化 TokenArray
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        # 创建迭代器
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        # 重置迭代计数器
        self.diff_iteration.iter_no = 0
        # 创建 KV cache
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        # 遍历每个块进行解码
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            # 解码当前块
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            # 如果所有序列都已完成解码，退出循环
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class IterationSmooth(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            # update KV-cache
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if kv_cache is None:
            logits = model(inputs_embeds=self.inputs_embeds).logits
            decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:block_loc.end] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
        self.num_forwards += 1
        self.iter_no += 1

class IterSmoothDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.diff_iteration = IterationSmooth(self.model, cont_weight, cont_weight_init, cont_weight_growth, threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates
    
    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        self.diff_iteration.reset_input_embeds(x)
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class VicinityCacheIteration(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, prefix_look, after_look, warmup_steps):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        if self.iter_no < self.warmup_steps:
            out_full = model(x.data)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(x.data, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        window_input = x.data[:, left_start:right_end]
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(window_input, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
        self.num_forwards += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x)
        self.iter_no += 1

class VicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with Vicinity Cache Update.

    The decoding algorithm defines a window to update KV-cache in each diffusion iteration.
    The window can be larger than the decoding block.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = VicinityCacheIteration(prefix_look, after_look, warmup_steps)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class IterSmoothWithVicinityCache(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, prefix_look, after_look, warmup_steps,
            cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)

        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0
    
    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.iter_no < self.warmup_steps:
            out_full = model(inputs_embeds=self.inputs_embeds)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(
                inputs_embeds=self.inputs_embeds[:, left_start:right_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
        )

        self.num_forwards += 1
        self.iter_no += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
        mask_index = (x.data[:, left_start:right_end] == decoder.mask_id)
        self.inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

class IterSmoothWithVicinityCacheDiffusionLLM(IterSmoothDiffusionLLM):
    """ This diffusion LLM inference generates tokens with vicinity cache and iteration smoothing.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3,
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = IterSmoothWithVicinityCache(model, prefix_look, after_look, warmup_steps,
                cont_weight=cont_weight, cont_weight_init=cont_weight_init, cont_weight_growth=cont_weight_growth,
                threshold_decay=threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates


class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[:, (self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[:, self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()

class BlockDiffusionLLMAttnmask(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of Attention Mask.

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        assert prompt.shape[0] == 1, "We currently only support batch size = 1."
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length
        
        
        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0)


        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        # We don't need kv_cache for the implementation of attention mask
        kv_cache = None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, 
                pos_ids, bd_attn_mask)
            if decode_compl:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class BlockDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of KV-Cache

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.cache_factory = cache_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        self.early_stop = early_stop

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        batch_size = prompt.shape[0]
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length

        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0).repeat(batch_size, 1)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        prompt_length = it._get_first_block_start()
        kv_cache = self.cache_factory.create()

        # prefill for kv_cache
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        prefill_length = max(prefill_length, block_length)
        self.block_runner.prefill(self.model, x[:, :prefill_length], kv_cache, pos_ids[:, :prefill_length], bd_attn_mask[:,:prefill_length,:prefill_length])
        
        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, pos_ids, bd_attn_mask)
            if torch.all(decode_compl) and self.early_stop:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()



class BlockMCMCDiffusionLLM(BlockWiseDiffusionLLM):
    """BlockWise Diffusion LLM with MCMC refinement (Power Sampling)

    This class extends BlockWiseDiffusionLLM to add MCMC-based block refinement
    using the Power Sampling algorithm. After each block is denoised, it performs
    MCMC iterations to sample from the power distribution p^α.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes tokens from logits
    iterator_factory : IteratorFactory
        Factory class that generates iterator on token array
    enable_mcmc : bool
        Whether to enable MCMC refinement (default: True)
    n_mcmc_steps : int
        Number of MCMC iterations per block (default: 5)
    mcmc_alpha : float
        Power parameter α for target distribution p^α (default: 4.0)
    mcmc_temperature : float
        Temperature for proposal distribution (default: 0.0)
    """
    def __init__(self, model, decoder, iterator_factory,
                 enable_mcmc=True, n_mcmc_steps=5,
                 mcmc_alpha=4.0, mcmc_temperature=0.0,
                 tokenizer=None, verbose=True, **kwargs):
        super().__init__(model, decoder, iterator_factory, **kwargs)
        self.enable_mcmc = enable_mcmc
        self.n_mcmc_steps = n_mcmc_steps
        self.mcmc_alpha = mcmc_alpha
        self.mcmc_temperature = mcmc_temperature
        self.tokenizer = tokenizer
        self.verbose = verbose

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        """Generate tokens with diffusion iterations and MCMC refinement"""
        # Initialize token array
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)

        # Initialize confidence tensors for MCMC
        confidences_norm = torch.full(x.data.shape, -np.inf, dtype=torch.float32, device=x.device)
        confidences_unnorm = torch.full(x.data.shape, -np.inf, dtype=torch.float32, device=x.device)

        # Create iterator
        it = self.iterator_factory.create(x, block_length)
        self.diff_iteration.iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None

        # Iterate over blocks
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            # Phase 1: Denoise block
            x, confidences_norm, confidences_unnorm = self._denoise_block(
                x, block_loc, confidences_norm, confidences_unnorm, kv_cache, block_id
            )

            # Phase 2: MCMC refinement
            if self.enable_mcmc:
                x, confidences_norm, confidences_unnorm, acceptance_rate = self._mcmc_refine_block(
                    x, block_loc, confidences_norm, confidences_unnorm, kv_cache
                )
                logger.info(f'Block {block_id} MCMC acceptance rate: {acceptance_rate:.2%}')

            # Early stop if EOS
            decode_compl = torch.any(x[:, block_loc.start:block_loc.end] == self.decoder.eos_id, dim=1)
            if torch.all(decode_compl):
                break

        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

    def _denoise_block(self, x, block_loc, confidences_norm, confidences_unnorm, kv_cache, block_id):
        """Denoise a block using iterative denoising with confidence tracking"""
        block = x[:, block_loc.start:block_loc.end]
        step_count = 0
        total_masks = (block == self.decoder.mask_id).sum().item()

        while (block == self.decoder.mask_id).sum() > 0:
            step_count += 1
            # Forward pass
            cache_update_kv = None
            if kv_cache is not None and kv_cache.require_update(self.diff_iteration.iter_no, block_loc.start, block_loc.end):
                output = self.model(x.data, use_cache=True)
                cache_update_kv = output.past_key_values
                self.diff_iteration.num_forwards += 1
                logits = output.logits[:, block_loc.start:block_loc.end]
                kv_cache.update(output.past_key_values)
                self.diff_iteration.cache_updates += 1
            elif kv_cache is None:
                logits = self.model(x.data).logits[:, block_loc.start:block_loc.end]
            elif kv_cache.cache_type == 'prefix':
                past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                logits = self.model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                        replace_position=replace_position).logits
                block_length = block_loc.end - block_loc.start
                logits = logits[:, :block_length]
            else:
                past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                logits = self.model(block, past_key_values=past_key_values, use_cache=True,
                        replace_position=replace_position).logits

            # Call decoder to decode and get confidences
            conf_norm_block, conf_unnorm_block = self.decoder.decode(
                logits, block_loc.start, block_loc.end, x, mcmc_alpha=self.mcmc_alpha
            )

            # Update global confidence tensors (only update non-inf values)
            mask_updated = conf_norm_block > -np.inf
            if mask_updated.any():
                confidences_norm[:, block_loc.start:block_loc.end][mask_updated] = conf_norm_block[mask_updated]
                confidences_unnorm[:, block_loc.start:block_loc.end][mask_updated] = conf_unnorm_block[mask_updated]

            block = x[:, block_loc.start:block_loc.end]
            self.diff_iteration.num_forwards += 1
            self.diff_iteration.iter_no += 1

            # Print intermediate denoising results
            if self.verbose and self.tokenizer is not None:
                prompt_length = x.prompt.shape[1]
                current_output = x.data[:, prompt_length:]
                decoded_output = self.tokenizer.batch_decode(current_output, skip_special_tokens=True)
                print(f"[_denoise_block] Block {block_id}, Step {step_count}/{total_masks}: {decoded_output}")
        # if self.verbose:
        #     print(f"confidences_norm:{confidences_norm}")
        #     print(f"confidences_unnorm:{confidences_unnorm}")

        return x, confidences_norm, confidences_unnorm

    def _mcmc_refine_block(self, x, block_loc, confidences_norm, confidences_unnorm, kv_cache):
        """Refine current block using MCMC (Metropolis-Hastings)"""
        acceptances = 0
        attempts = 0
        prompt_length = x.prompt.shape[1]

        for mcmc_step in range(self.n_mcmc_steps):
            attempts += 1

            # Step 1: Randomly select resampling position within current block (must be >= prompt_length)
            idx = random.randint(max(block_loc.start, prompt_length), block_loc.end - 1)

            # Step 2: Generate proposal sequence
            x_prop, conf_norm_prop, conf_unnorm_prop = self._generate_proposal(
                x, idx, block_loc.end, confidences_norm.clone(), confidences_unnorm.clone()
            )

            # Step 3: Compute MH acceptance ratio
            log_r = self._compute_acceptance_ratio(
                confidences_norm, confidences_unnorm,
                conf_norm_prop, conf_unnorm_prop,
                idx, block_loc.end
            )

            # Step 4: Accept/reject decision
            accept_prob = min(1.0, np.exp(min(log_r, 0.0)))
            accepted = np.random.rand() < accept_prob

            if self.verbose and self.tokenizer is not None:
                print(f"\n[MCMC Step {mcmc_step+1}/{self.n_mcmc_steps}] idx={idx}, log_r={log_r:.4f}, accept_prob={accept_prob:.4f}, accepted={accepted}")

            if accepted:
                acceptances += 1
                x = x_prop
                # 只更新重新采样区域的置信度
                confidences_norm[:, idx:block_loc.end] = conf_norm_prop[:, idx:block_loc.end]
                confidences_unnorm[:, idx:block_loc.end] = conf_unnorm_prop[:, idx:block_loc.end]

                if self.verbose and self.tokenizer is not None:
                    current_output = x.data[:, prompt_length:]
                    decoded = self.tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[ACCEPTED] New sequence: {decoded}")
                    # print(f"new_confidences_norm:{confidences_norm}")
                    # print(f"new_confidences_unnorm:{confidences_unnorm}")
            else:
                if self.verbose and self.tokenizer is not None:
                    current_output = x.data[:, prompt_length:]
                    decoded = self.tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[REJECTED] Keep sequence: {decoded}")

        # Check for EOS token after MCMC refinement
        if self.decoder.eos_id in x.data[0, block_loc.start:block_loc.end]:
            # Find EOS position
            block_tokens = x.data[0, block_loc.start:block_loc.end].tolist()
            eos_idx = block_tokens.index(self.decoder.eos_id) + block_loc.start

            # Truncate confidences at EOS
            confidences_norm[:, eos_idx+1:] = -np.inf
            confidences_unnorm[:, eos_idx+1:] = -np.inf

            if self.verbose:
                print(f"[MCMC] EOS token detected at position {eos_idx}, truncating confidences")

        acceptance_rate = acceptances / attempts if attempts > 0 else 0.0
        return x, confidences_norm, confidences_unnorm, acceptance_rate

    def _generate_proposal(self, x_current, idx, block_end, confidences_norm, confidences_unnorm):
        """Generate proposal sequence by remasking and denoising from idx to block_end"""
        # Clone current sequence
        x_prop = TokenArray(x_current.prompt, x_current.gen_length, self.decoder.mask_id, self.decoder.eos_id, x_current.device)
        x_prop.data = x_current.data.clone()

        # Remask from idx to block_end
        x_prop.data[:, idx:block_end] = self.decoder.mask_id

        # Reset confidences for remasked region
        conf_norm_prop = confidences_norm.clone()
        conf_unnorm_prop = confidences_unnorm.clone()
        conf_norm_prop[:, idx:block_end] = -np.inf
        conf_unnorm_prop[:, idx:block_end] = -np.inf

        # Iterative denoising (same as _denoise_block but for proposal)
        block = x_prop.data[:, idx:block_end]
        block_mask_index = (block == self.decoder.mask_id)
        steps = (block == self.decoder.mask_id).sum().item()

        if self.verbose and self.tokenizer is not None:
            print(f"[_generate_proposal] Starting denoising from idx={idx} to {block_end}, steps={steps}")

        if steps > 0:
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                # Forward pass (no KV cache for simplicity)
                logits = self.model(x_prop.data).logits[:, idx:block_end]

                # Compute dual log probabilities
                log_p_norm = torch.nn.functional.log_softmax(logits, dim=-1)
                log_p_unnorm = torch.nn.functional.log_softmax(self.mcmc_alpha * logits, dim=-1)

                # Sample with Gumbel noise
                logits_with_noise = add_gumbel_noise_power(logits, alpha=1.0, temperature=self.mcmc_temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Get log probs for selected tokens
                x0_logp_norm = torch.gather(log_p_norm, -1, x0.unsqueeze(-1)).squeeze(-1)
                x0_logp_unnorm = torch.gather(log_p_unnorm, -1, x0.unsqueeze(-1)).squeeze(-1)

                # Compute confidence for remasking
                p = torch.nn.functional.softmax(logits, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                # Select tokens to transfer
                mask_index = (block == self.decoder.mask_id)
                x0 = torch.where(mask_index, x0, block)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True

                # Update proposal sequence and confidences
                if transfer_index.any():
                    x_prop.data[:, idx:block_end][transfer_index] = x0[transfer_index]
                    conf_norm_prop[:, idx:block_end][transfer_index] = x0_logp_norm[transfer_index].float()
                    conf_unnorm_prop[:, idx:block_end][transfer_index] = x0_logp_unnorm[transfer_index].float()

                block = x_prop.data[:, idx:block_end]

                # Print intermediate proposal generation results
                if self.verbose and self.tokenizer is not None:
                    prompt_length = x_prop.prompt.shape[1]
                    current_output = x_prop.data[:, prompt_length:block_end]
                    decoded_proposal = self.tokenizer.batch_decode(current_output, skip_special_tokens=True)
                    print(f"[_generate_proposal] Step {i+1}/{steps}: {decoded_proposal}")

        return x_prop, conf_norm_prop, conf_unnorm_prop

    def _compute_acceptance_ratio(self, confidences_norm_cur, confidences_unnorm_cur,
                                   confidences_norm_prop, confidences_unnorm_prop,
                                   idx, block_end):
        """Compute Metropolis-Hastings acceptance ratio"""
        # Extract log probabilities for the resampled region [idx, block_end)
        log_prob_cur_norm = confidences_norm_cur[:, idx:block_end].view(-1).tolist()
        log_prob_cur_unnorm = confidences_unnorm_cur[:, idx:block_end].view(-1).tolist()
        log_prob_prop_norm = confidences_norm_prop[:, idx:block_end].view(-1).tolist()
        log_prob_prop_unnorm = confidences_unnorm_prop[:, idx:block_end].view(-1).tolist()

        # Filter out -inf values
        log_prob_cur_norm = [x for x in log_prob_cur_norm if x > -np.inf]
        log_prob_cur_unnorm = [x for x in log_prob_cur_unnorm if x > -np.inf]
        log_prob_prop_norm = [x for x in log_prob_prop_norm if x > -np.inf]
        log_prob_prop_unnorm = [x for x in log_prob_prop_unnorm if x > -np.inf]

        print(f"[Acceptance Ratio] log_prob_cur_norm: {log_prob_cur_norm} | sum: {sum(log_prob_cur_norm)}")
        print(f"[Acceptance Ratio] log_prob_cur_unnorm: {log_prob_cur_unnorm} | sum: {sum(log_prob_cur_unnorm)}")
        print(f"[Acceptance Ratio] log_prob_prop_norm: {log_prob_prop_norm} | sum: {sum(log_prob_prop_norm)}")
        print(f"[Acceptance Ratio] log_prob_prop_unnorm: {log_prob_prop_unnorm} | sum: {sum(log_prob_prop_unnorm)}")

        assert len(log_prob_cur_norm) == len(log_prob_prop_norm), "Mismatched lengths in norm log probs"
        assert len(log_prob_cur_unnorm) == len(log_prob_prop_unnorm), "Mismatched lengths in unnorm log probs"
        
        # MH acceptance ratio: log r = log[p^α(x') * q(x|x')] - log[p^α(x) * q(x'|x)]
        log_r = (sum(log_prob_prop_unnorm) + sum(log_prob_cur_norm)
                - sum(log_prob_cur_unnorm) - sum(log_prob_prop_norm))

        return log_r