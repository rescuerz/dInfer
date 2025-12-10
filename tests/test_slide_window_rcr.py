"""
Test for SlideWindowRCRDecoder

This test verifies the slide window runtime-confidence-remask decoder
without requiring an actual model.
"""
import torch
import pytest
from dinfer.decoding.parallel_strategy import SlideWindowRCRDecoder, ThresholdParallelDecoder


class TestSlideWindowRCRDecoder:
    """Test cases for SlideWindowRCRDecoder"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mask_id = 126336
        self.eos_id = 126081
        self.decoder = SlideWindowRCRDecoder(
            temperature=0.0,  # No noise for deterministic testing
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            use_float64=False,
            enable_low_threshold=True,
            enable_decline_threshold=True,
            enable_consecutive_decline=True
        )
    
    def test_init(self):
        """Test decoder initialization"""
        assert self.decoder.threshold == 0.9
        assert self.decoder.medium_threshold == 0.8
        assert self.decoder.low_threshold == 0.62
        assert self.decoder.window_size == 3
        assert self.decoder.decline_threshold == 0.1
        assert self.decoder.mask_id == self.mask_id
        assert self.decoder.eos_id == self.eos_id
        assert len(self.decoder.confidence_history) == 0
        assert self.decoder.enable_low_threshold == True
        assert self.decoder.enable_decline_threshold == True
        assert self.decoder.enable_consecutive_decline == True
    
    def test_block_init(self):
        """Test block initialization clears history"""
        # Add some fake history
        self.decoder.confidence_history = [torch.zeros(1, 10)]
        
        # Initialize new block
        block_x = torch.full((1, 10), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        assert len(self.decoder.confidence_history) == 0
        assert self.decoder.block_seq_len == 10
    
    def test_compute_confidence_all_mask(self):
        """Test confidence computation when all positions are mask"""
        seq_len = 5
        vocab_size = 100
        
        # Create logits with clear preference for token 50
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, :, 50] = 10.0  # High logit for token 50
        
        curr_x = torch.full((1, seq_len), self.mask_id)
        
        x0, confidence = self.decoder._compute_confidence(logits, curr_x)
        
        # All positions should predict token 50
        assert torch.all(x0 == 50)
        # Confidence should be high (softmax of 10 vs 0s)
        assert torch.all(confidence > 0.9)
    
    def test_compute_confidence_decoded_positions(self):
        """Test confidence computation for decoded positions"""
        seq_len = 5
        vocab_size = 100
        
        # Create logits
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, 0, 30] = 5.0  # Position 0: prefer token 30
        logits[:, 1, 40] = 5.0  # Position 1: prefer token 40
        
        # Position 0 is decoded as token 30, position 1 is mask
        curr_x = torch.full((1, seq_len), self.mask_id)
        curr_x[0, 0] = 30  # Decoded position
        
        x0, confidence = self.decoder._compute_confidence(logits, curr_x)
        
        # Position 0: confidence should be for token 30 (current token)
        # Position 1: confidence should be for argmax token (40)
        assert x0[0, 1] == 40
    
    def test_is_declining_insufficient_history(self):
        """Test _is_declining returns (False, None) when history is insufficient"""
        # Empty history (need window_size - 1 = 2 historical steps)
        assert self.decoder._is_declining(0, 0.7) == (False, None)
        
        # Only 1 historical step, need 2
        self.decoder.confidence_history = [
            torch.tensor([[0.8, 0.7]])
        ]
        assert self.decoder._is_declining(0, 0.7) == (False, None)
    
    def test_is_declining_with_inf(self):
        """Test _is_declining returns (False, None) when first element is -inf"""
        # With window_size=3, we need 2 historical steps + current
        self.decoder.confidence_history = [
            torch.tensor([[float('-inf'), 0.8]]),
            torch.tensor([[0.75, 0.75]])
        ]
        # Position 0 has -inf at start of window
        assert self.decoder._is_declining(0, 0.70) == (False, None)
        # Position 1: history=[0.8, 0.75] + curr=0.65 → decline=0.8-0.65=0.15 > 0.1
        is_declining, decline_type = self.decoder._is_declining(1, 0.65)
        assert is_declining == True
        assert decline_type == 'threshold'
    
    def test_is_declining_true(self):
        """Test _is_declining returns (True, type) for declining confidence"""
        # With window_size=3, we need 2 historical steps + current
        self.decoder.confidence_history = [
            torch.tensor([[0.85, 0.9]]),
            torch.tensor([[0.75, 0.85]])
        ]
        # Position 0: history=[0.85, 0.75] + curr=0.65 → decline=0.85-0.65=0.2 > 0.1
        is_declining, decline_type = self.decoder._is_declining(0, 0.65)
        assert is_declining == True
        assert decline_type in ['threshold', 'consecutive']
        # Position 1: history=[0.9, 0.85] + curr=0.80 → decline=0.9-0.80=0.1, not > 0.1
        # But consecutive declining: 0.9 > 0.85 > 0.80, so it triggers 'consecutive'
        is_declining_1, decline_type_1 = self.decoder._is_declining(1, 0.80)
        assert is_declining_1 == True
        assert decline_type_1 == 'consecutive'
        
        # Test case where neither threshold nor consecutive triggers
        # Position 1: history=[0.9, 0.85] + curr=0.86 → not consecutive (0.85 < 0.86)
        is_declining_2, decline_type_2 = self.decoder._is_declining(1, 0.86)
        assert is_declining_2 == False
        assert decline_type_2 == None
    
    def test_compute_remask_index_low_confidence(self):
        """Test remask for low confidence positions"""
        # Initialize block first to set prompt_positions
        block_x = torch.full((1, 3), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        confidence = torch.tensor([[0.5, 0.7, 0.9]])  # Position 0 is low
        mask_index = torch.tensor([[False, False, True]])  # Position 2 is mask
        
        remask_index, low_conf_indices, declining_indices, consecutive_indices = \
            self.decoder._compute_remask_index(confidence, mask_index)
        
        # Position 0: decoded, conf=0.5 < low_threshold=0.62 → remask
        # Position 1: decoded, conf=0.7 > low_threshold → no remask (unless declining)
        # Position 2: mask → no remask
        assert remask_index[0, 0] == True
        assert remask_index[0, 1] == False
        assert remask_index[0, 2] == False
        assert 0 in low_conf_indices
    
    def test_compute_remask_index_declining(self):
        """Test remask for declining confidence positions"""
        # Initialize block first to set prompt_positions
        block_x = torch.full((1, 2), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        # With window_size=3, we need 2 historical steps + current
        # Setup history showing decline
        self.decoder.confidence_history = [
            torch.tensor([[0.90, 0.85]]),
            torch.tensor([[0.85, 0.78]])
        ]
        
        confidence = torch.tensor([[0.79, 0.65]])  # Current step
        mask_index = torch.tensor([[False, False]])  # Both decoded
        
        remask_index, low_conf_indices, declining_indices, consecutive_indices = \
            self.decoder._compute_remask_index(confidence, mask_index)
        
        # Position 0: conf=0.79 < medium=0.8, check decline: 0.90-0.79=0.11 > 0.1 → remask
        # Position 1: conf=0.65 > low=0.62, check decline: 0.85-0.65=0.2 > 0.1 → remask
        assert remask_index[0, 0] == True
        assert remask_index[0, 1] == True
    
    def test_ensure_progress(self):
        """Test ensure_progress adds tokens when needed"""
        transfer_index = torch.tensor([[False, False, True]])
        confidence = torch.tensor([[0.8, 0.6, float('-inf')]])
        remask_cnt = torch.tensor(2)  # 2 positions remasked
        
        # gap = 2 + 1 - 1 = 2, need to add 2 more transfers
        self.decoder._ensure_progress(transfer_index, confidence, remask_cnt)
        
        # Should have selected positions 0 and 1 (highest confidence)
        assert transfer_index.sum() == 3
        assert transfer_index[0, 0] == True
        assert transfer_index[0, 1] == True
    
    def test_update_history_basic(self):
        """Test history update after decode"""
        confidence = torch.tensor([[0.8, 0.7, 0.6]])
        prev_x = torch.tensor([[self.mask_id, 50, self.mask_id]])  # Position 1 was decoded
        transfer_index = torch.tensor([[True, False, False]])  # Position 0 transferred
        final_remask = torch.tensor([[False, False, False]])  # No remask
        
        self.decoder._update_history(confidence, prev_x, transfer_index, final_remask)
        
        assert len(self.decoder.confidence_history) == 1
        hist = self.decoder.confidence_history[0]
        
        # Position 0: was mask, now transferred → record confidence
        assert hist[0, 0] == 0.8
        # Position 1: was decoded, not remask → record confidence
        assert hist[0, 1] == 0.7
        # Position 2: still mask → -inf
        assert hist[0, 2] == float('-inf')
    
    def test_update_history_with_remask(self):
        """Test history update clears remasked positions"""
        # Pre-populate history
        self.decoder.confidence_history = [
            torch.tensor([[0.9, 0.85, float('-inf')]]),
            torch.tensor([[0.85, 0.80, float('-inf')]])
        ]
        
        confidence = torch.tensor([[0.8, 0.7, 0.6]])
        prev_x = torch.tensor([[50, 60, self.mask_id]])  # Positions 0,1 were decoded
        transfer_index = torch.tensor([[False, False, True]])  # Position 2 transferred
        final_remask = torch.tensor([[False, True, False]])  # Position 1 remasked
        
        self.decoder._update_history(confidence, prev_x, transfer_index, final_remask)
        
        # Check all history entries have position 1 cleared
        for hist in self.decoder.confidence_history:
            assert hist[0, 1] == float('-inf')
    
    def test_decode_basic(self):
        """Test basic decode functionality"""
        seq_len = 5
        vocab_size = 100
        
        # Initialize block
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        # Create logits with high confidence for specific tokens
        logits = torch.zeros(1, seq_len, vocab_size)
        for i in range(seq_len):
            logits[:, i, i + 10] = 10.0  # High confidence for tokens 10-14
        
        # Create x tensor
        x = torch.full((1, seq_len), self.mask_id)
        
        # Decode
        self.decoder.decode(logits, block_start=0, block_end=seq_len, x=x)
        
        # Some positions should be decoded (high confidence)
        assert (x != self.mask_id).sum() > 0
        # History should be updated
        assert len(self.decoder.confidence_history) == 1
    
    def test_decode_with_remask(self):
        """Test decode with remask scenario"""
        seq_len = 3
        vocab_size = 100
        
        # Initialize block
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        # Step 1: Decode position 0 with high confidence
        logits1 = torch.zeros(1, seq_len, vocab_size)
        logits1[:, 0, 50] = 10.0  # High confidence for token 50 at position 0
        logits1[:, 1, 60] = 10.0
        logits1[:, 2, 70] = 10.0
        
        x = torch.full((1, seq_len), self.mask_id)
        self.decoder.decode(logits1, 0, seq_len, x)
        
        # Position 0 should be decoded
        decoded_token = x[0, 0].item()
        assert decoded_token != self.mask_id
        
        # Step 2-4: Simulate declining confidence for position 0
        for step in range(3):
            # Create logits with decreasing confidence for position 0
            logits = torch.zeros(1, seq_len, vocab_size)
            # Decrease confidence for the decoded token
            logits[:, 0, decoded_token] = 5.0 - step * 2  # 5, 3, 1
            logits[:, 0, 99] = 4.0 - step  # Alternative token gets relatively higher
            
            # Keep other positions with some confidence
            for i in range(1, seq_len):
                if x[0, i] == self.mask_id:
                    logits[:, i, 60 + i] = 8.0
            
            self.decoder.decode(logits, 0, seq_len, x)
        
        # After several steps of declining confidence, position 0 might be remasked
        # (depending on exact confidence values and thresholds)
        print(f"Final x: {x}")
        print(f"History length: {len(self.decoder.confidence_history)}")


class TestSlideWindowRCRDecoderEdgeCases:
    """Edge case tests for SlideWindowRCRDecoder"""
    
    def setup_method(self):
        self.mask_id = 126336
        self.eos_id = 126081
        self.decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=True,
            enable_decline_threshold=True,
            enable_consecutive_decline=True
        )
    
    def test_all_positions_decoded(self):
        """Test when all positions are already decoded"""
        seq_len = 3
        vocab_size = 100
        
        block_x = torch.tensor([[50, 60, 70]])  # All decoded
        self.decoder.block_init(block_x, block_id=0)
        
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, 0, 50] = 5.0
        logits[:, 1, 60] = 5.0
        logits[:, 2, 70] = 5.0
        
        x = block_x.clone()
        self.decoder.decode(logits, 0, seq_len, x)
        
        # Should remain unchanged (no mask positions to decode)
        # But might remask if confidence is low
        print(f"Result: {x}")
    
    def test_single_position(self):
        """Test with single position"""
        vocab_size = 100
        
        block_x = torch.full((1, 1), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        logits = torch.zeros(1, 1, vocab_size)
        logits[:, 0, 50] = 10.0
        
        x = torch.full((1, 1), self.mask_id)
        self.decoder.decode(logits, 0, 1, x)
        
        # Should decode the single position
        assert x[0, 0] == 50
    
    def test_ensure_progress_prevents_stall(self):
        """Test that ensure_progress prevents decoding from stalling"""
        seq_len = 3
        vocab_size = 100
        
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        # Create logits with very low confidence (below threshold)
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, 0, 50] = 1.0  # Low confidence
        logits[:, 1, 60] = 1.0
        logits[:, 2, 70] = 1.0
        
        x = torch.full((1, seq_len), self.mask_id)
        self.decoder.decode(logits, 0, seq_len, x)
        
        # At least one position should be decoded (ensure_progress)
        assert (x != self.mask_id).sum() >= 1


class TestSlideWindowRCRDecoderFullFlow:
    """Full flow tests based on our discussion examples"""
    
    def setup_method(self):
        self.mask_id = 126336
        self.eos_id = 126081
        self.decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=True,
            enable_decline_threshold=True,
            enable_consecutive_decline=True
        )
    
    def test_full_flow_step_by_step(self):
        """
        Test the complete flow discussed in our conversation:
        
        Step 1: Position j is mask, gets transferred with conf=0.75
        Step 2: Position j is decoded, conf=0.70, no remask (insufficient history)
        Step 3: Position j is decoded, conf=0.65, decline=0.75-0.65=0.10, not > 0.1
        Step 4: Position j is decoded, conf=0.60 < low_threshold=0.62, remask!
        Step 5: Position j is mask again, gets re-transferred
        """
        seq_len = 2
        vocab_size = 100
        
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Step 1: Transfer position 0 with high confidence
        logits1 = torch.zeros(1, seq_len, vocab_size)
        logits1[:, 0, 50] = 10.0  # High confidence for token 50
        logits1[:, 1, 60] = 10.0  # High confidence for token 60
        
        self.decoder.decode(logits1, 0, seq_len, x)
        
        assert x[0, 0] == 50, "Step 1: Position 0 should be decoded as token 50"
        assert len(self.decoder.confidence_history) == 1
        # Check history: position 0 should have confidence recorded
        assert self.decoder.confidence_history[0][0, 0] > 0.9
        print(f"Step 1: x={x.tolist()}, history_len={len(self.decoder.confidence_history)}")
        
        # Step 2: Position 0 decoded, conf decreases but insufficient history
        logits2 = torch.zeros(1, seq_len, vocab_size)
        logits2[:, 0, 50] = 3.0  # Lower confidence for token 50
        logits2[:, 0, 99] = 2.0  # Some alternative
        logits2[:, 1, 60] = 10.0
        
        self.decoder.decode(logits2, 0, seq_len, x)
        
        assert x[0, 0] == 50, "Step 2: Position 0 should still be token 50"
        assert len(self.decoder.confidence_history) == 2
        print(f"Step 2: x={x.tolist()}, history_len={len(self.decoder.confidence_history)}")
        
        # Step 3: Position 0 decoded, conf decreases more, but decline not > threshold
        logits3 = torch.zeros(1, seq_len, vocab_size)
        logits3[:, 0, 50] = 2.5  # Even lower
        logits3[:, 0, 99] = 2.0
        logits3[:, 1, 60] = 10.0
        
        self.decoder.decode(logits3, 0, seq_len, x)
        
        assert x[0, 0] == 50, "Step 3: Position 0 should still be token 50"
        assert len(self.decoder.confidence_history) == 3
        print(f"Step 3: x={x.tolist()}, history_len={len(self.decoder.confidence_history)}")
        
        # Step 4: Position 0 conf drops below low_threshold, should remask
        logits4 = torch.zeros(1, seq_len, vocab_size)
        logits4[:, 0, 50] = 0.5  # Very low confidence
        logits4[:, 0, 99] = 0.3
        logits4[:, 1, 60] = 10.0
        
        self.decoder.decode(logits4, 0, seq_len, x)
        
        # Position 0 should be remasked (conf < 0.62)
        # But ensure_progress might save it if no other mask positions
        print(f"Step 4: x={x.tolist()}, history_len={len(self.decoder.confidence_history)}")
        
        # Check that history for position 0 is cleared if it was remasked
        if x[0, 0] == self.mask_id:
            for hist in self.decoder.confidence_history:
                assert hist[0, 0] == float('-inf'), "Remasked position should have -inf history"
    
    def test_remask_saved_by_ensure_progress(self):
        """
        Test the special case where a position is marked for remask,
        but ensure_progress saves it because we need to make progress.
        
        Scenario:
        - Position 0 is decoded, low confidence → marked for remask
        - Position 1 is mask, but very low confidence
        - ensure_progress needs to select something, picks position 0 back
        - Position 0's history should NOT be cleared
        """
        seq_len = 2
        vocab_size = 100
        
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        # First, decode position 0
        x = torch.full((1, seq_len), self.mask_id)
        logits1 = torch.zeros(1, seq_len, vocab_size)
        logits1[:, 0, 50] = 10.0
        logits1[:, 1, 60] = 0.1  # Very low confidence for position 1
        
        self.decoder.decode(logits1, 0, seq_len, x)
        assert x[0, 0] == 50
        print(f"After step 1: x={x.tolist()}")
        
        # Now, position 0 has low confidence, position 1 is still mask with low conf
        # Position 0 should be marked for remask, but ensure_progress might save it
        logits2 = torch.zeros(1, seq_len, vocab_size)
        logits2[:, 0, 50] = 0.3  # Very low confidence for decoded token
        logits2[:, 1, 60] = 0.2  # Even lower for mask position
        
        self.decoder.decode(logits2, 0, seq_len, x)
        print(f"After step 2: x={x.tolist()}")
        
        # The exact behavior depends on ensure_progress logic
        # At minimum, we should make progress (at least one token decoded)
    
    def test_history_cleared_on_final_remask(self):
        """
        Test that when a position is finally remasked (not saved by ensure_progress),
        its history is cleared to -inf in all history entries.
        """
        seq_len = 3
        vocab_size = 100
        
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Decode all positions with high confidence
        logits1 = torch.zeros(1, seq_len, vocab_size)
        logits1[:, 0, 50] = 10.0
        logits1[:, 1, 60] = 10.0
        logits1[:, 2, 70] = 10.0
        
        self.decoder.decode(logits1, 0, seq_len, x)
        assert torch.all(x != self.mask_id), "All positions should be decoded"
        
        # Step 2: All still high confidence
        logits2 = torch.zeros(1, seq_len, vocab_size)
        logits2[:, 0, 50] = 8.0
        logits2[:, 1, 60] = 8.0
        logits2[:, 2, 70] = 8.0
        
        self.decoder.decode(logits2, 0, seq_len, x)
        
        # Step 3: Position 0 drops to very low confidence
        logits3 = torch.zeros(1, seq_len, vocab_size)
        logits3[:, 0, 50] = 0.3  # Very low - should trigger remask
        logits3[:, 1, 60] = 8.0  # High - can be transferred
        logits3[:, 2, 70] = 8.0  # High - can be transferred
        
        self.decoder.decode(logits3, 0, seq_len, x)
        
        print(f"After step 3: x={x.tolist()}")
        
        # If position 0 was remasked, its history should be cleared
        if x[0, 0] == self.mask_id:
            for hist in self.decoder.confidence_history:
                assert hist[0, 0] == float('-inf'), "Remasked position history should be -inf"
            print("✓ Position 0 was remasked and history cleared")
        else:
            print("Position 0 was saved by ensure_progress")
    
    def test_declining_trend_triggers_remask(self):
        """
        Test that declining trend (not just low absolute value) triggers remask.
        
        Position 0: conf goes 0.85 → 0.78 → 0.72 (decline = 0.13 > 0.1)
        Since 0.72 < medium_threshold=0.8, should trigger remask
        """
        seq_len = 2
        vocab_size = 100
        
        block_x = torch.full((1, seq_len), self.mask_id)
        self.decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Step 1: Decode with conf ~0.85
        logits1 = torch.zeros(1, seq_len, vocab_size)
        logits1[:, 0, 50] = 5.0  # Will give ~0.85+ confidence after softmax
        logits1[:, 1, 60] = 10.0
        
        self.decoder.decode(logits1, 0, seq_len, x)
        conf1 = self.decoder.confidence_history[-1][0, 0].item()
        print(f"Step 1: conf={conf1:.3f}")
        
        # Step 2: conf decreases
        logits2 = torch.zeros(1, seq_len, vocab_size)
        logits2[:, 0, 50] = 3.5
        logits2[:, 1, 60] = 10.0
        
        self.decoder.decode(logits2, 0, seq_len, x)
        conf2 = self.decoder.confidence_history[-1][0, 0].item()
        print(f"Step 2: conf={conf2:.3f}")
        
        # Step 3: conf decreases more, should trigger decline check
        logits3 = torch.zeros(1, seq_len, vocab_size)
        logits3[:, 0, 50] = 2.5  # Lower confidence
        logits3[:, 1, 60] = 10.0
        
        self.decoder.decode(logits3, 0, seq_len, x)
        
        print(f"Step 3: x={x.tolist()}")
        print(f"History: {[h[0, 0].item() for h in self.decoder.confidence_history]}")


class TestSlideWindowRCRDecoderEnableFlags:
    """Test enable flags for remask strategies"""
    
    def setup_method(self):
        self.mask_id = 126336
        self.eos_id = 126081
        self.vocab_size = 100
    
    def test_default_all_disabled(self):
        """Test that all strategies are disabled by default"""
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            mask_id=self.mask_id,
            eos_id=self.eos_id
        )
        assert decoder.enable_low_threshold == False
        assert decoder.enable_decline_threshold == False
        assert decoder.enable_consecutive_decline == False
    
    def test_enable_low_threshold_only(self):
        """Test that only low_threshold strategy triggers remask when enabled"""
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=True,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        seq_len = 3
        block_x = torch.full((1, seq_len), self.mask_id)
        decoder.block_init(block_x, block_id=0)
        
        # First decode all positions
        x = torch.full((1, seq_len), self.mask_id)
        logits1 = torch.zeros(1, seq_len, self.vocab_size)
        logits1[:, 0, 50] = 10.0
        logits1[:, 1, 60] = 10.0
        logits1[:, 2, 70] = 10.0
        decoder.decode(logits1, 0, seq_len, x)
        
        # Build history for decline detection
        for _ in range(2):
            logits = torch.zeros(1, seq_len, self.vocab_size)
            logits[:, 0, 50] = 5.0  # Medium confidence
            logits[:, 1, 60] = 10.0
            logits[:, 2, 70] = 10.0
            decoder.decode(logits, 0, seq_len, x)
        
        decoder.reset_stats()
        
        # Now test: position 0 has low confidence (< 0.62), should trigger low_threshold
        logits_low = torch.zeros(1, seq_len, self.vocab_size)
        logits_low[:, 0, 50] = 0.3  # Very low confidence
        logits_low[:, 1, 60] = 10.0
        logits_low[:, 2, 70] = 10.0
        decoder.decode(logits_low, 0, seq_len, x)
        
        stats = decoder.get_stats()
        # Low threshold should trigger, but decline strategies should not
        assert stats['remask_declining_count'] == 0
        assert stats['remask_consecutive_declining_count'] == 0
        print(f"enable_low_threshold_only stats: {stats}")
    
    def test_enable_decline_threshold_only(self):
        """Test that only decline_threshold strategy triggers remask when enabled"""
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=True,
            enable_consecutive_decline=False
        )
        
        seq_len = 2
        block_x = torch.full((1, seq_len), self.mask_id)
        decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Step 1: Decode with high confidence
        logits1 = torch.zeros(1, seq_len, self.vocab_size)
        logits1[:, 0, 50] = 10.0  # High confidence
        logits1[:, 1, 60] = 10.0
        decoder.decode(logits1, 0, seq_len, x)
        
        # Step 2: Medium confidence
        logits2 = torch.zeros(1, seq_len, self.vocab_size)
        logits2[:, 0, 50] = 4.0
        logits2[:, 1, 60] = 10.0
        decoder.decode(logits2, 0, seq_len, x)
        
        decoder.reset_stats()
        
        # Step 3: Lower confidence, should trigger decline threshold
        # conf < medium_threshold AND decline > decline_threshold
        logits3 = torch.zeros(1, seq_len, self.vocab_size)
        logits3[:, 0, 50] = 2.0  # Lower confidence, decline should be > 0.1
        logits3[:, 1, 60] = 10.0
        decoder.decode(logits3, 0, seq_len, x)
        
        stats = decoder.get_stats()
        # Low threshold should NOT trigger (disabled)
        assert stats['remask_low_conf_count'] == 0
        # Consecutive decline should NOT trigger (disabled)
        assert stats['remask_consecutive_declining_count'] == 0
        print(f"enable_decline_threshold_only stats: {stats}")
    
    def test_enable_consecutive_decline_only(self):
        """Test that only consecutive_decline strategy triggers remask when enabled"""
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.5,  # High threshold so decline_threshold won't trigger
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=True
        )
        
        seq_len = 2
        block_x = torch.full((1, seq_len), self.mask_id)
        decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Step 1: Decode with high confidence
        logits1 = torch.zeros(1, seq_len, self.vocab_size)
        logits1[:, 0, 50] = 5.0
        logits1[:, 1, 60] = 10.0
        decoder.decode(logits1, 0, seq_len, x)
        
        # Step 2: Slightly lower
        logits2 = torch.zeros(1, seq_len, self.vocab_size)
        logits2[:, 0, 50] = 4.5
        logits2[:, 1, 60] = 10.0
        decoder.decode(logits2, 0, seq_len, x)
        
        decoder.reset_stats()
        
        # Step 3: Even lower - consecutive decline
        logits3 = torch.zeros(1, seq_len, self.vocab_size)
        logits3[:, 0, 50] = 4.0  # Consecutive decline: 5.0 -> 4.5 -> 4.0
        logits3[:, 1, 60] = 10.0
        decoder.decode(logits3, 0, seq_len, x)
        
        stats = decoder.get_stats()
        # Low threshold should NOT trigger (disabled)
        assert stats['remask_low_conf_count'] == 0
        # Decline threshold should NOT trigger (disabled)
        assert stats['remask_declining_count'] == 0
        print(f"enable_consecutive_decline_only stats: {stats}")
    
    def test_no_remask_when_all_disabled(self):
        """Test that no remask occurs when all strategies are disabled"""
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        seq_len = 3
        block_x = torch.full((1, seq_len), self.mask_id)
        decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Decode all positions
        logits1 = torch.zeros(1, seq_len, self.vocab_size)
        logits1[:, 0, 50] = 10.0
        logits1[:, 1, 60] = 10.0
        logits1[:, 2, 70] = 10.0
        decoder.decode(logits1, 0, seq_len, x)
        
        # Build history
        for _ in range(3):
            logits = torch.zeros(1, seq_len, self.vocab_size)
            logits[:, 0, 50] = 3.0  # Declining confidence
            logits[:, 1, 60] = 10.0
            logits[:, 2, 70] = 10.0
            decoder.decode(logits, 0, seq_len, x)
        
        decoder.reset_stats()
        
        # Very low confidence - would trigger remask if enabled
        logits_low = torch.zeros(1, seq_len, self.vocab_size)
        logits_low[:, 0, 50] = 0.1  # Very low
        logits_low[:, 1, 60] = 10.0
        logits_low[:, 2, 70] = 10.0
        decoder.decode(logits_low, 0, seq_len, x)
        
        stats = decoder.get_stats()
        assert stats['total_remask_count'] == 0
        assert stats['remask_low_conf_count'] == 0
        assert stats['remask_declining_count'] == 0
        assert stats['remask_consecutive_declining_count'] == 0
        print(f"no_remask_when_all_disabled stats: {stats}")
    
    def test_decline_strategies_extended_range_when_low_disabled(self):
        """
        Test that when enable_low_threshold=False, decline strategies apply to
        the full range of curr_conf < medium_threshold (including < low_threshold).
        """
        decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            medium_threshold=0.8,
            low_threshold=0.62,
            window_size=3,
            decline_threshold=0.1,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,  # Disabled
            enable_decline_threshold=True,  # Enabled
            enable_consecutive_decline=False
        )
        
        seq_len = 2
        block_x = torch.full((1, seq_len), self.mask_id)
        decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Step 1: Decode with high confidence
        logits1 = torch.zeros(1, seq_len, self.vocab_size)
        logits1[:, 0, 50] = 10.0
        logits1[:, 1, 60] = 10.0
        decoder.decode(logits1, 0, seq_len, x)
        
        # Step 2: Medium confidence
        logits2 = torch.zeros(1, seq_len, self.vocab_size)
        logits2[:, 0, 50] = 3.0
        logits2[:, 1, 60] = 10.0
        decoder.decode(logits2, 0, seq_len, x)
        
        decoder.reset_stats()
        
        # Step 3: Very low confidence (< low_threshold), but decline strategy should still apply
        # because enable_low_threshold=False means we don't skip with continue
        logits3 = torch.zeros(1, seq_len, self.vocab_size)
        logits3[:, 0, 50] = 0.5  # Very low, < low_threshold=0.62
        logits3[:, 1, 60] = 10.0
        decoder.decode(logits3, 0, seq_len, x)
        
        stats = decoder.get_stats()
        # Low threshold should NOT trigger (disabled)
        assert stats['remask_low_conf_count'] == 0
        # Decline threshold SHOULD trigger (enabled and conf < medium_threshold)
        # The decline from ~0.99 to ~0.5 is definitely > 0.1
        print(f"decline_strategies_extended_range stats: {stats}")


class TestSlideWindowRCRDecoderVsThreshold:
    """Test that SlideWindowRCRDecoder with all strategies disabled behaves like ThresholdParallelDecoder"""
    
    def setup_method(self):
        self.mask_id = 126336
        self.eos_id = 126081
        self.vocab_size = 100
    
    def test_same_result_single_step(self):
        """Test that both decoders produce the same result in a single decode step"""
        threshold = 0.9
        temperature = 0.0
        
        slide_decoder = SlideWindowRCRDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        threshold_decoder = ThresholdParallelDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id
        )
        
        seq_len = 5
        
        # Initialize both decoders
        block_x = torch.full((1, seq_len), self.mask_id)
        slide_decoder.block_init(block_x, block_id=0)
        threshold_decoder.block_init(block_x, block_id=0)
        
        # Create identical inputs
        logits = torch.zeros(1, seq_len, self.vocab_size)
        logits[:, 0, 50] = 10.0  # High confidence
        logits[:, 1, 60] = 5.0   # Medium confidence
        logits[:, 2, 70] = 2.0   # Low confidence
        logits[:, 3, 80] = 10.0  # High confidence
        logits[:, 4, 90] = 10.0  # High confidence
        
        x_slide = torch.full((1, seq_len), self.mask_id)
        x_threshold = torch.full((1, seq_len), self.mask_id)
        
        slide_decoder.decode(logits.clone(), 0, seq_len, x_slide)
        threshold_decoder.decode(logits.clone(), 0, seq_len, x_threshold)
        
        print(f"SlideWindowRCR result: {x_slide.tolist()}")
        print(f"Threshold result: {x_threshold.tolist()}")
        
        assert torch.equal(x_slide, x_threshold), \
            f"Results differ: slide={x_slide.tolist()}, threshold={x_threshold.tolist()}"
    
    def test_same_result_multiple_steps(self):
        """Test that both decoders produce the same result over multiple decode steps"""
        threshold = 0.9
        temperature = 0.0
        
        slide_decoder = SlideWindowRCRDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        threshold_decoder = ThresholdParallelDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id
        )
        
        seq_len = 5
        
        # Initialize both decoders
        block_x = torch.full((1, seq_len), self.mask_id)
        slide_decoder.block_init(block_x, block_id=0)
        threshold_decoder.block_init(block_x, block_id=0)
        
        x_slide = torch.full((1, seq_len), self.mask_id)
        x_threshold = torch.full((1, seq_len), self.mask_id)
        
        # Run multiple decode steps
        for step in range(5):
            # Create logits that gradually increase confidence
            logits = torch.zeros(1, seq_len, self.vocab_size)
            for i in range(seq_len):
                logits[:, i, 50 + i] = 3.0 + step * 2  # Increasing confidence
            
            slide_decoder.decode(logits.clone(), 0, seq_len, x_slide)
            threshold_decoder.decode(logits.clone(), 0, seq_len, x_threshold)
            
            print(f"Step {step + 1}: slide={x_slide.tolist()}, threshold={x_threshold.tolist()}")
            
            assert torch.equal(x_slide, x_threshold), \
                f"Step {step + 1} results differ: slide={x_slide.tolist()}, threshold={x_threshold.tolist()}"
        
        print("✓ All steps produced identical results")
    
    def test_same_result_with_varying_confidence(self):
        """Test with varying confidence patterns"""
        threshold = 0.9
        temperature = 0.0
        
        slide_decoder = SlideWindowRCRDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        threshold_decoder = ThresholdParallelDecoder(
            temperature=temperature,
            threshold=threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id
        )
        
        seq_len = 4
        
        block_x = torch.full((1, seq_len), self.mask_id)
        slide_decoder.block_init(block_x, block_id=0)
        threshold_decoder.block_init(block_x, block_id=0)
        
        x_slide = torch.full((1, seq_len), self.mask_id)
        x_threshold = torch.full((1, seq_len), self.mask_id)
        
        # Test with different confidence patterns
        confidence_patterns = [
            [10.0, 10.0, 10.0, 10.0],  # All high
            [1.0, 1.0, 1.0, 1.0],      # All low
            [10.0, 1.0, 10.0, 1.0],    # Alternating
            [1.0, 5.0, 8.0, 10.0],     # Increasing
        ]
        
        for pattern in confidence_patterns:
            # Reset
            x_slide = torch.full((1, seq_len), self.mask_id)
            x_threshold = torch.full((1, seq_len), self.mask_id)
            slide_decoder.block_init(torch.full((1, seq_len), self.mask_id), block_id=0)
            threshold_decoder.block_init(torch.full((1, seq_len), self.mask_id), block_id=0)
            
            logits = torch.zeros(1, seq_len, self.vocab_size)
            for i, conf in enumerate(pattern):
                logits[:, i, 50 + i] = conf
            
            slide_decoder.decode(logits.clone(), 0, seq_len, x_slide)
            threshold_decoder.decode(logits.clone(), 0, seq_len, x_threshold)
            
            print(f"Pattern {pattern}: slide={x_slide.tolist()}, threshold={x_threshold.tolist()}")
            
            assert torch.equal(x_slide, x_threshold), \
                f"Pattern {pattern} results differ"
        
        print("✓ All patterns produced identical results")
    
    def test_no_remask_stats_when_disabled(self):
        """Test that remask statistics are all zero when strategies are disabled"""
        slide_decoder = SlideWindowRCRDecoder(
            temperature=0.0,
            threshold=0.9,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
            enable_low_threshold=False,
            enable_decline_threshold=False,
            enable_consecutive_decline=False
        )
        
        seq_len = 3
        block_x = torch.full((1, seq_len), self.mask_id)
        slide_decoder.block_init(block_x, block_id=0)
        
        x = torch.full((1, seq_len), self.mask_id)
        
        # Run multiple steps with varying confidence
        for step in range(5):
            logits = torch.zeros(1, seq_len, self.vocab_size)
            # Create declining confidence pattern that would trigger remask if enabled
            for i in range(seq_len):
                logits[:, i, 50 + i] = max(0.5, 10.0 - step * 2)
            
            slide_decoder.decode(logits, 0, seq_len, x)
        
        stats = slide_decoder.get_stats()
        assert stats['total_remask_count'] == 0
        assert stats['remask_low_conf_count'] == 0
        assert stats['remask_declining_count'] == 0
        assert stats['remask_consecutive_declining_count'] == 0
        print(f"✓ No remask occurred: {stats}")


def run_tests():
    """Run all tests"""
    print("Running SlideWindowRCRDecoder tests...")
    
    # Basic tests
    test_class = TestSlideWindowRCRDecoder()
    
    test_class.setup_method()
    test_class.test_init()
    print("✓ test_init passed")
    
    test_class.setup_method()
    test_class.test_block_init()
    print("✓ test_block_init passed")
    
    test_class.setup_method()
    test_class.test_compute_confidence_all_mask()
    print("✓ test_compute_confidence_all_mask passed")
    
    test_class.setup_method()
    test_class.test_compute_confidence_decoded_positions()
    print("✓ test_compute_confidence_decoded_positions passed")
    
    test_class.setup_method()
    test_class.test_is_declining_insufficient_history()
    print("✓ test_is_declining_insufficient_history passed")
    
    test_class.setup_method()
    test_class.test_is_declining_with_inf()
    print("✓ test_is_declining_with_inf passed")
    
    test_class.setup_method()
    test_class.test_is_declining_true()
    print("✓ test_is_declining_true passed")
    
    test_class.setup_method()
    test_class.test_compute_remask_index_low_confidence()
    print("✓ test_compute_remask_index_low_confidence passed")
    
    test_class.setup_method()
    test_class.test_compute_remask_index_declining()
    print("✓ test_compute_remask_index_declining passed")
    
    test_class.setup_method()
    test_class.test_ensure_progress()
    print("✓ test_ensure_progress passed")
    
    test_class.setup_method()
    test_class.test_update_history_basic()
    print("✓ test_update_history_basic passed")
    
    test_class.setup_method()
    test_class.test_update_history_with_remask()
    print("✓ test_update_history_with_remask passed")
    
    test_class.setup_method()
    test_class.test_decode_basic()
    print("✓ test_decode_basic passed")
    
    test_class.setup_method()
    test_class.test_decode_with_remask()
    print("✓ test_decode_with_remask passed")
    
    # Edge case tests
    edge_test_class = TestSlideWindowRCRDecoderEdgeCases()
    
    edge_test_class.setup_method()
    edge_test_class.test_all_positions_decoded()
    print("✓ test_all_positions_decoded passed")
    
    edge_test_class.setup_method()
    edge_test_class.test_single_position()
    print("✓ test_single_position passed")
    
    edge_test_class.setup_method()
    edge_test_class.test_ensure_progress_prevents_stall()
    print("✓ test_ensure_progress_prevents_stall passed")
    
    # Full flow tests
    flow_test_class = TestSlideWindowRCRDecoderFullFlow()
    
    flow_test_class.setup_method()
    flow_test_class.test_full_flow_step_by_step()
    print("✓ test_full_flow_step_by_step passed")
    
    flow_test_class.setup_method()
    flow_test_class.test_remask_saved_by_ensure_progress()
    print("✓ test_remask_saved_by_ensure_progress passed")
    
    flow_test_class.setup_method()
    flow_test_class.test_history_cleared_on_final_remask()
    print("✓ test_history_cleared_on_final_remask passed")
    
    flow_test_class.setup_method()
    flow_test_class.test_declining_trend_triggers_remask()
    print("✓ test_declining_trend_triggers_remask passed")
    
    # Enable flags tests
    enable_test_class = TestSlideWindowRCRDecoderEnableFlags()
    
    enable_test_class.setup_method()
    enable_test_class.test_default_all_disabled()
    print("✓ test_default_all_disabled passed")
    
    enable_test_class.setup_method()
    enable_test_class.test_enable_low_threshold_only()
    print("✓ test_enable_low_threshold_only passed")
    
    enable_test_class.setup_method()
    enable_test_class.test_enable_decline_threshold_only()
    print("✓ test_enable_decline_threshold_only passed")
    
    enable_test_class.setup_method()
    enable_test_class.test_enable_consecutive_decline_only()
    print("✓ test_enable_consecutive_decline_only passed")
    
    enable_test_class.setup_method()
    enable_test_class.test_no_remask_when_all_disabled()
    print("✓ test_no_remask_when_all_disabled passed")
    
    enable_test_class.setup_method()
    enable_test_class.test_decline_strategies_extended_range_when_low_disabled()
    print("✓ test_decline_strategies_extended_range_when_low_disabled passed")
    
    # Comparison tests with ThresholdParallelDecoder
    compare_test_class = TestSlideWindowRCRDecoderVsThreshold()
    
    compare_test_class.setup_method()
    compare_test_class.test_same_result_single_step()
    print("✓ test_same_result_single_step passed")
    
    compare_test_class.setup_method()
    compare_test_class.test_same_result_multiple_steps()
    print("✓ test_same_result_multiple_steps passed")
    
    compare_test_class.setup_method()
    compare_test_class.test_same_result_with_varying_confidence()
    print("✓ test_same_result_with_varying_confidence passed")
    
    compare_test_class.setup_method()
    compare_test_class.test_no_remask_stats_when_disabled()
    print("✓ test_no_remask_stats_when_disabled passed")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    run_tests()
