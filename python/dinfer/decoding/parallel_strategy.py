from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise, get_num_transfer_tokens

@ torch.no_grad()
@ torch.compile(dynamic=True)
def get_transfer_index_hierarchy_fast_v2(logits, temperature, remasking, mask_index, x, num_transfer_tokens,  mask_id, threshold=None,  low_threshold = None):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
    
    else:
        for i in range (mask_index.shape[0]):

            mask_i = mask_index[i].int()
            conf_i = confidence[i]

            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index [i, max_index] = True
                    continue


            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]


            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True
            
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold) 
                
        if threshold is not None:
            transfer_index = torch.logical_or(transfer_index, confidence > threshold)


    return x0, transfer_index

@ torch.no_grad()
def get_transfer_index_hierarchy_remask(logits, temperature, mask_index, x, num_transfer_tokens,  
                                         mask_id, threshold=None,  low_threshold = None, remask_threshold = 0.4):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l


    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l


    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and (lower_index, torch.logical_not(mask_index))
    mask_new = torch.logical_or (lower_index, mask_index)

    
    confidence = torch.where(mask_new, x0_p, float('-inf'))
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    remask_cnt = remask_index.sum (dim = 1)

    
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
    
    else:
        for i in range (mask_new.shape[0]):

            mask_i = mask_new[i].int()
            conf_i = confidence[i]


            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]


            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True
            
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold) 
                
            if threshold is not None:
                transfer_index [i] = torch.logical_or(transfer_index [i], conf_i > threshold)

            gap = int((remask_cnt [i] + 1 - transfer_index [i].sum()).item())
            if gap > 0:
                conf_i [transfer_index [i]] = float('-inf')
                values, indices = torch.topk (conf_i, gap, largest=True, sorted=False)
                transfer_index [i][indices] = True
            
    
    remask_index = torch.logical_and (remask_index, torch.logical_not (transfer_index))
    x0 [remask_index] = mask_id
    transfer_index [remask_index] = True

    return x0, transfer_index


def get_transfer_index_cache (logits, mask_index, x, block_end, num_transfer_tokens, temperature, remasking, threshold=None, minimal_topk=1):

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(logits.dtype)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0[mask_index], -1)), -1)  # b, l
        confidence = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # print("num_transfer_tokens, topk",num_transfer_tokens[0], minimal_topk)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(self, temperature, remasking='low_confidence', mask_id=126336):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """

# Parallel decoding only
@ torch.compile(dynamic=True)
def get_transfer_index_threshold(logits, temperature, mask_index, x, mask_id,
        threshold, rm_mask=True, use_float64=False, **kwargs):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    
    # gurantee the denoised token will not be the mask_id   
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    actual_threshold = (torch.max(confidence, dim=1)[0]-1e-5).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index

class ThresholdParallelDecoder(ParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold.

    The decoder decodes a token when its confidence score is larger than a threshold.
    """
    def __init__(self, temperature, threshold, remasking='low_confidence', mask_id=126336, eos_id=126081,
            use_float64=False):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, use_float64=self.use_float64)
        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

class CreditThresholdParallelDecoder(ThresholdParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold + credit.

    The decoder decodes a token when its confidence is larger than a threshold.
    """
    def __init__(self, 
                 credit_alpha=0.7, 
                 boost_gamma=0.2, 
                 decay_beta=0.8,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.credit_alpha = credit_alpha
        self.boost_gamma = boost_gamma
        self.decay_beta = decay_beta

        self._credit_mats = {}   
        self._credit_iters = {}  

    def _apply_credit_fusion(self, logits, mask_index, key):
        """
        EMA-based credit fusion (no CM, no pre-credit):
        - Maintains a per-block CreditMatrix (EMA with decay).
        - Accumulates enhanced top-1 probability only on masked positions.
        - Returns fused_logits.
        """
        B, L, V = logits.shape
        device = logits.device

        mat = self._credit_mats.get(key, None)
        if mat is None or mat.shape != (B, L, V) or mat.device != device:
            mat = torch.zeros((B, L, V), dtype=torch.float32, device=device)
            self._credit_mats[key] = mat
            self._credit_iters[key] = 0

        iter_idx = self._credit_iters[key]

        if iter_idx > 0:
            mat.mul_(self.decay_beta)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top1_probs, top1_idx = torch.max(probs, dim=-1)         
        enhanced = top1_probs.pow(self.boost_gamma).to(mat.dtype)  
        update_vals = enhanced * mask_index.to(enhanced.dtype)     
        mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

        fused_logits = logits + self.credit_alpha * torch.log(mat + 1)
        self._credit_iters[key] = iter_idx + 1
        return fused_logits

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key)

        x0, transfer_index = get_transfer_index_threshold(used_logits, self.temperature, mask_index, curr_x,
                self.mask_id, threshold=iter_threshold, use_float64=self.use_float64)

        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

        if hasattr(x, 'data'):
            has_mask = (x.data == self.mask_id).any()
        else:
            has_mask = (x == self.mask_id).any() if x.dim() > 0 else (x == self.mask_id)

        if not has_mask:
            self._credit_mats.clear()
            self._credit_iters.clear()

class FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=126336):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]


class HierarchyDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a hierarchy way. Forcing LLMs to decode tokens seperately.
    """
    def __init__(self, temperature, remasking='low_confidence',
                mask_id=126336,  eos_id=126081, 
                threshold=None, low_threshold=0.4):
        super().__init__(temperature, remasking, mask_id)
        self.iter = 0
        self.mask_id = mask_id
        self.eos_id=eos_id
        self.threshold=threshold
        self.low_threshold=low_threshold

    def get_transfer_index(self, logits,  mask_index, iter_threshold, **kwargs):
    
        B, L = mask_index.shape

        # TODO(DuLun): support batch size > 1
        assert B == 1

        device = logits.device
        
        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
        else:
            logits_with_noise = logits

        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
        
        x0_logp = F.log_softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p = x0_logp.exp()  # b, l

        neg_inf_val = torch.finfo(x0_p.dtype).min
        confidence = torch.where(mask_index, x0_p, torch.tensor(neg_inf_val, device=device, dtype=x0_p.dtype))
        
        prev = torch.cat(
            [mask_index.new_zeros((B, 1), dtype=torch.bool), mask_index[:, :-1]],
            dim=1
        )
        starts = torch.logical_and(mask_index, torch.logical_not(prev))

        seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
        seg_id = torch.where(mask_index, seg_id, 0)

        seg_max = torch.full((B, L), neg_inf_val, device=device, dtype=confidence.dtype)
        seg_max = torch.scatter_reduce(seg_max, dim=1, index=seg_id, src=confidence, reduce='amax', include_self=True)

        seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
        transfer_index = (confidence == seg_max_at_pos)

        if self.low_threshold is not None:
            transfer_index = torch.logical_and(transfer_index, torch.gt(confidence, self.low_threshold))
        if iter_threshold is not None:
            transfer_index = torch.logical_or(transfer_index, torch.gt(confidence, iter_threshold))

        
        top1_idx = torch.argmax(confidence, dim=-1)
        top1 = torch.nn.functional.one_hot(top1_idx, num_classes=L).to(torch.bool)
        transfer_index = torch.logical_or(transfer_index, top1)
        

        return x0, transfer_index

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == self.mask_id
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = self.get_transfer_index(logits, mask_index, iter_threshold)
        self.iter += 1
        transfer_index = torch.logical_and(transfer_index, mask_index)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]


class SlideWindowRCRDecoder(ParallelDecoder):
    """ Slide Window Runtime-Confidence-Remask Decoder.
    
    This decoder tracks confidence history over a sliding window of decoding steps.
    It remasks decoded tokens when:
    1. Current confidence < low_threshold (absolute low)
    2. Current confidence < medium_threshold AND confidence is declining over the window
    
    Parameters
    ----------
    temperature : float
        Temperature for Gumbel noise sampling
    threshold : float
        High confidence threshold for transfer decision (default: 0.9)
    medium_threshold : float
        Medium confidence upper bound, used with decline detection (default: 0.8)
    low_threshold : float
        Low confidence threshold, directly remask (default: 0.62)
    window_size : int
        Size of sliding window for confidence history (default: 3)
    decline_threshold : float
        Threshold for confidence decline detection (default: 0.1)
    mask_id : int
        Token ID for mask
    eos_id : int
        Token ID for end of sequence
    use_float64 : bool
        Whether to use float64 for softmax computation
    """
    
    def __init__(self, temperature, threshold=0.9, medium_threshold=0.8, 
                 low_threshold=0.62, window_size=3, decline_threshold=0.1,
                 mask_id=126336, eos_id=126081, use_float64=False):
        super().__init__(temperature, 'low_confidence', mask_id)
        self.threshold = threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.window_size = window_size
        self.decline_threshold = decline_threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64
        
        # Confidence history: list of tensors, each shape [1, seq_len]
        # -inf means the position is mask or not yet tracked
        self.confidence_history = []
        self.block_seq_len = None
    
    def block_init(self, block_x, block_id):
        """ Initialize for a new block. Clear confidence history.
        """
        self.confidence_history = []
        self.block_seq_len = block_x.shape[1]
    
    def _compute_confidence(self, logits, curr_x):
        """ Compute confidence for all positions.
        
        For mask positions: use argmax token's probability
        For decoded positions: use current token's probability
        
        Returns
        -------
        x0 : Tensor
            Predicted tokens (argmax of logits with noise)
        confidence : Tensor
            Confidence scores for all positions
        """
        B, L, V = logits.shape
        
        # Add Gumbel noise for sampling
        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(logits, temperature=self.temperature)
        else:
            logits_with_noise = logits
        
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]
        
        # Compute softmax probabilities
        if self.use_float64:
            p = F.softmax(logits.to(torch.float64), dim=-1)
        else:
            p = F.softmax(logits.to(torch.float32), dim=-1)
        
        # For mask positions, use x0's probability
        # For decoded positions, use current token's probability
        mask_index = (curr_x == self.mask_id)
        target_tokens = torch.where(mask_index, x0, curr_x)
        confidence = torch.gather(p, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        
        return x0, confidence
    
    def _is_declining(self, pos_idx):
        """ Check if confidence at position pos_idx is declining over the window.
        
        Returns True if:
        - Window is full (len >= window_size)
        - First element is not -inf
        - decline = history[0] - history[-1] > decline_threshold
        """
        if len(self.confidence_history) < self.window_size:
            return False
        
        # Get history for this position from the last window_size steps
        history = [self.confidence_history[-(self.window_size - i)][0, pos_idx].item() 
                   for i in range(self.window_size)]
        
        # If first is -inf, cannot compute valid decline
        if history[0] == float('-inf'):
            return False
        
        decline = history[0] - history[-1]
        return decline > self.decline_threshold
    
    def _compute_remask_index(self, confidence, mask_index):
        """ Compute which decoded positions should be remasked.
        
        Remask conditions (only for decoded positions):
        1. confidence < low_threshold (absolute low)
        2. confidence < medium_threshold AND _is_declining (trend-based)
        """
        B, L = confidence.shape
        remask_index = torch.zeros_like(mask_index, dtype=torch.bool)
        
        for j in range(L):
            # Only consider decoded positions (not mask)
            if mask_index[0, j]:
                continue
            
            curr_conf = confidence[0, j].item()
            
            # Condition 1: absolute low confidence
            if curr_conf < self.low_threshold:
                remask_index[0, j] = True
                continue
            
            # Condition 2: medium confidence + declining trend
            if curr_conf < self.medium_threshold:
                if self._is_declining(j):
                    remask_index[0, j] = True
        
        return remask_index
    
    def _ensure_progress(self, transfer_index, confidence, remask_cnt):
        """ Ensure at least one token is decoded per step.
        
        If remask_cnt + 1 > transfer_cnt, select additional tokens by confidence.
        """
        gap = int((remask_cnt + 1 - transfer_index.sum()).item())
        if gap > 0:
            # Set already selected positions to -inf
            conf_for_select = confidence.clone()
            conf_for_select[transfer_index] = float('-inf')
            
            # Select top-gap positions by confidence
            _, indices = torch.topk(conf_for_select.view(-1), gap, largest=True, sorted=False)
            transfer_index.view(-1)[indices] = True
    
    def _update_history(self, confidence, prev_x, transfer_index, final_remask):
        """ Update confidence history after all decisions are made.
        
        Rules:
        - final_remask positions: set to -inf (clear history)
        - decoded positions (including newly transferred): record confidence
        - mask positions: set to -inf
        """
        B, L = confidence.shape
        new_history = torch.full_like(confidence, float('-inf'))
        
        # Positions that were mask before this step
        was_mask = (prev_x == self.mask_id)
        
        # Positions that are decoded after this step (either already decoded or newly transferred)
        # and not final_remask
        is_decoded = torch.logical_or(
            torch.logical_not(was_mask),  # was already decoded
            transfer_index  # newly transferred
        )
        is_decoded_and_not_remask = torch.logical_and(is_decoded, torch.logical_not(final_remask))
        
        # Record confidence for decoded positions
        new_history[is_decoded_and_not_remask] = confidence[is_decoded_and_not_remask]
        
        # Append to history, maintain window size
        self.confidence_history.append(new_history.clone())
        if len(self.confidence_history) > self.window_size:
            self.confidence_history.pop(0)
        
        # Clear history for final_remask positions in all history entries
        if final_remask.any():
            for hist in self.confidence_history:
                hist[final_remask] = float('-inf')
    
    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block with slide window remask.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        
        B, L = logits.shape[0], logits.shape[1]
        assert B == 1, "SlideWindowRCRDecoder only supports batch_size=1"
        
        curr_x = x[:, block_start:block_end]
        mask_index = (curr_x == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]
        
        # 1. Compute confidence for all positions
        x0, confidence = self._compute_confidence(logits, curr_x)
        
        # 2. Compute remask index (only for decoded positions)
        remask_index = self._compute_remask_index(confidence, mask_index)
        
        # 3. Compute new mask region (original mask + remask)
        mask_new = torch.logical_or(mask_index, remask_index)
        
        # 4. Compute transfer_index using threshold strategy
        # Similar to get_transfer_index_threshold but on mask_new
        confidence_for_transfer = torch.where(mask_new, confidence, torch.tensor(float('-inf'), device=confidence.device))
        
        # Ensure denoised token is not mask_id
        mask_new_and_valid = mask_new & (x0 != self.mask_id)
        x0_safe = torch.where(mask_new_and_valid, x0, curr_x)
        confidence_for_transfer = torch.where(mask_new_and_valid, confidence, torch.tensor(float('-inf'), device=confidence.device))
        
        # Apply threshold: transfer if confidence >= threshold
        # But at least transfer the highest confidence token
        actual_threshold = (torch.max(confidence_for_transfer, dim=1)[0] - 1e-5).clamp(-1000, iter_threshold).unsqueeze(-1)
        transfer_index = confidence_for_transfer >= actual_threshold
        
        # 5. Ensure progress: at least decode one token per step
        remask_cnt = remask_index.sum()
        self._ensure_progress(transfer_index, confidence_for_transfer, remask_cnt)
        
        # 6. Compute final remask (remask but not transferred)
        final_remask = torch.logical_and(remask_index, torch.logical_not(transfer_index))
        
        # 7. Update x0 for final_remask positions
        x0[final_remask] = self.mask_id
        
        # 8. Apply transfer: only transfer within mask_new region
        transfer_index = torch.logical_and(transfer_index, mask_new)
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        
        # 9. Update confidence history (after all decisions)
        self._update_history(confidence, curr_x, transfer_index, final_remask)
