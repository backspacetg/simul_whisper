import os
import logging

import torch
import torch.nn.functional as F

from ..whisper import load_model, DecodingOptions, tokenizer
from .config import AlignAttConfig
from ..whisper.audio import log_mel_spectrogram, TOKENS_PER_SECOND, pad_or_trim, N_SAMPLES, N_FRAMES
from ..whisper.timing import median_filter
from ..whisper.decoding import SuppressBlank, GreedyDecoder, SuppressTokens
import os

DEC_PAD = 50257
logger = logging.getLogger(__name__)

class PaddedAlignAttWhisper:
    def __init__(self, cfg: AlignAttConfig) -> None:
            
        model_name = os.path.basename(cfg.model_path).replace(".pt", "")
        model_path = os.path.dirname(cfg.model_path)
        self.model = load_model(name=model_name, download_root=model_path)
        checkpoint = torch.load(cfg.if_ckpt_path)
        self.CIFLinear = torch.nn.Linear(self.model.dims.n_audio_state, 1)
        self.CIFLinear.load_state_dict(checkpoint)
        self.CIFLinear.to(self.model.device)

        decode_options = DecodingOptions(
            language = cfg.language, 
            without_timestamps = True,
            task="transcribe"
        )
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=True, 
            language=cfg.language, 
            task=decode_options.task
        )
        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)
        self.cfg = cfg
        # install hooks
        self.dec_attns = []
        def layer_hook(module, net_input, net_output):
            # net_output[1]: B*num_head*token_len*audio_len
            t = F.softmax(net_output[1], dim=-1)
            self.dec_attns.append(t.squeeze(0))
        for b in self.model.decoder.blocks:
            b.cross_attn.register_forward_hook(layer_hook)
        
        self.kv_cache = {}
        def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
            if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
                # save as-is, for the first token or cross attention
                self.kv_cache[module.cache_id] = net_output
            else:
                self.kv_cache[module.cache_id] = torch.cat([self.kv_cache[module.cache_id], net_output], dim=1).detach()
            return self.kv_cache[module.cache_id] 

        for b in self.model.decoder.blocks:
            b.attn.key.register_forward_hook(kv_hook)
            b.attn.value.register_forward_hook(kv_hook)
            b.cross_attn.key.register_forward_hook(kv_hook)
            b.cross_attn.value.register_forward_hook(kv_hook)

        self.align_source = {}
        self.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.align_source.get(layer_rank, [])
            heads.append((self.num_align_heads, head_id.item()))
            self.align_source[layer_rank] = heads
            self.num_align_heads += 1

        self.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence, 
            dtype=torch.long, 
            device=self.model.device).unsqueeze(0)
        self.initial_token_length = self.initial_tokens.shape[1]
        self.tokens = [self.initial_tokens]
        self.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)

        suppress_tokens = [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
                # self.tokenizer.eot 
            ]
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens =  tuple(sorted(set(suppress_tokens)))

        self.logit_filters = []
        self.logit_filters.append(SuppressBlank(self.tokenizer, self.initial_token_length))
        self.logit_filters.append(SuppressTokens(suppress_tokens))
        self.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)

        self.segments = []
        self.new_segment = True

        self.last_attend_frame = -self.cfg.rewind_threshold
        self.drop_count = 0
        self.keep_count = 0

    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if not self.new_segment:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]
        logit = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
        return logit
    

    def refresh_segment(self, complete=False):
        if not complete and len(self.segments) > 2:
            self.tokens = [self.initial_tokens] 
            self.segments = self.segments[-2:]
            self.last_attend_frame = -self.cfg.rewind_threshold
        else:
            self.tokens = [self.initial_tokens]
            self.segments = []
            self.last_attend_frame = -self.cfg.rewind_threshold       


    # from https://github.com/dqqcasia/mosst/blob/master/fairseq/models/speech_to_text/convtransformer_wav2vec_cif.py
    def resize(self, alphas, target_lengths, threshold=0.999):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        # sum
        _num = alphas.sum(-1)
        num = target_lengths.float()
        # scaling
        _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))
        # rm attention value that exceeds threashold
        count = 0
        while len(torch.where(_alphas > threshold)[0]):
            count += 1
            if count > 10:
                break
            print('fixing alpha')
            xs, ys = torch.where(_alphas > threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num   
    
    
    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        content_mel_len = chunked_encoder_feature.shape[1] # B, T, D
        alphas = self.CIFLinear(chunked_encoder_feature).squeeze(dim=2) # B, T
        alphas = torch.sigmoid(alphas)
        decode_length = torch.round(alphas.sum(-1)).int()
        alphas, _ = self.resize(alphas, decode_length)
        alphas = alphas.squeeze(0) # (T, )
        threshold = 0.999
        integrate = torch.cumsum(alphas[:-1], dim=0) # ignore the peak value at the end of the content chunk
        exceed_count = integrate[-1] // threshold
        integrate = integrate - exceed_count*1.0 # minus 1 every time intergrate exceed the threshold
        important_positions = (integrate >= 0).nonzero(as_tuple=True)[0]
        if important_positions.numel() == 0:
            return False
        else:
            return important_positions[0] >= content_mel_len-2


    def infer(self, segment, is_last=False):
        self.new_segment = True
        with torch.no_grad():
            self.segments.append(segment)
            if len(self.segments) * self.cfg.segment_length < self.cfg.min_seg_len: 
                logger.debug("waiting for next segment")
                return self.initial_tokens.new_tensor([]), False
            if len(self.segments) * self.cfg.segment_length >= self.cfg.buffer_len:
                self.segments = self.segments[1:]
                self.tokens = [self.initial_tokens] + self.tokens[2:]
                self.last_attend_frame -= int(TOKENS_PER_SECOND*self.cfg.segment_length)
                logger.debug(f"remove segments: {len(self.segments)} {len(self.tokens)}")
            if len(self.segments) > 1:
                input_segments = torch.cat(self.segments, dim=0)
            else:
                input_segments = self.segments[0]
            if len(self.tokens) > 1:
                current_tokens = torch.cat(self.tokens, dim=1)
            else:
                current_tokens = self.tokens[0]
            
            mel_padded = log_mel_spectrogram(input_segments, padding=N_SAMPLES, device=self.model.device).unsqueeze(0)
            logger.debug(f"after padding: {mel_padded.shape}")
            mel = pad_or_trim(mel_padded, N_FRAMES)
            logger.debug(f"after trim {mel.shape}")
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2])/2)

            encoder_feature = self.model.encoder(mel)
            sum_logprobs = torch.zeros(1, device=mel.device)
            completed = False

            attn_of_alignment_heads = None
            token_len_before_decoding = current_tokens.shape[1]
            
            most_attened_frame = None
            fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])

            while not completed and current_tokens.shape[1] < self.max_text_len: # bos is 3 tokens

                logits = self.logits(current_tokens, encoder_feature) # B, len(tokens), token dict size

                if self.new_segment and self.tokenizer.no_speech is not None:
                    probs_at_sot = logits[:, self.sot_index, :].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                    if no_speech_probs[0] > self.cfg.nonspeech_prob:
                        break

                self.new_segment = False
                logits = logits[:, -1, :] # logits for the last token
                for logits_filter in self.logit_filters:
                    logits_filter.apply(logits, current_tokens)
                current_tokens, completed = self.token_decoder.update(current_tokens, logits, sum_logprobs)

                if completed:
                    logger.debug("decode stopped")

                attn_of_alignment_heads = [[] for _ in range(self.num_align_heads)]
                for i, attn_mat in enumerate(self.dec_attns):
                    layer_rank = int(i % len(self.model.decoder.blocks))
                    align_heads_in_layer = self.align_source.get(layer_rank, [])
                    if len(align_heads_in_layer) == 0:
                        continue
                    for align_head_rank, head_id in align_heads_in_layer:
                        attn_of_alignment_heads[align_head_rank].append(attn_mat[head_id, :, :])
                tmp = []
                for mat in attn_of_alignment_heads:
                    tmp.append(torch.cat(mat, dim=0))
                attn_of_alignment_heads = torch.stack(tmp, dim=0)
                std, mean = torch.std_mean(attn_of_alignment_heads, dim=-2, keepdim=True, unbiased=False)
                attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std
                attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7) # from whisper.timing
                attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=0)
                attn_of_alignment_heads = attn_of_alignment_heads[:, :content_mel_len]

                most_attened_frame = torch.argmax(attn_of_alignment_heads[-1, :], dim=0)

                if completed:
                    current_tokens = current_tokens[:, :-1]
                    break
                
                # for some rare cases where the attention fails
                if not is_last and self.last_attend_frame - most_attened_frame > self.cfg.rewind_threshold:
                    if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                        logger.debug("ommit rewinding from special tokens")
                        self.last_attend_frame = most_attened_frame
                    else:
                        logger.debug(f"[rewind detected] current attention pos: {most_attened_frame.item()}, last attention pos: {self.last_attend_frame.item()}; omit this segment")
                        self.last_attend_frame = -self.cfg.rewind_threshold
                        current_tokens = torch.cat(self.tokens, dim=1) if len(self.tokens) > 0 else self.tokens[0]
                        break
                else:
                    self.last_attend_frame = most_attened_frame

                if content_mel_len - most_attened_frame <= (4 if is_last else self.cfg.frame_threshold):
                    logger.debug(f"attention reaches the end: {most_attened_frame.item()}/{content_mel_len}")
                    current_tokens = current_tokens[:, :-1]
                    break
            
                logger.debug("attn: {}, current pos: {}, current token: {}({})".format(
                    attn_of_alignment_heads.shape if attn_of_alignment_heads is not None else None,
                    most_attened_frame, 
                    current_tokens[:, -1].item(),
                    self.tokenizer.decode([current_tokens[:, -1].item()])
                ))

            if attn_of_alignment_heads is not None:
                seg_len = int(self.cfg.segment_length*TOKENS_PER_SECOND)
                new_token_attn = attn_of_alignment_heads[token_len_before_decoding:, -seg_len:]
                if new_token_attn.shape[0] == 0:
                    logger.debug("no token generated")
                    logger.debug(f"token len {current_tokens.shape}")
                else:
                    new_token_max_attn, _ = new_token_attn.max(dim=1)
                    logger.debug(f"segment max attention: {new_token_max_attn.mean().item()/len(self.segments)}")

            new_tokens = current_tokens.new_tensor([]).unsqueeze(0)
            tokens_to_split = current_tokens[:, token_len_before_decoding:]
            if fire_detected or is_last:
                new_tokens = tokens_to_split
                self.keep_count += 1
            else:
                self.drop_count += 1
                tokens_to_split = tokens_to_split.squeeze(0)
                text_to_split = self.tokenizer.decode(tokens_to_split)
                logger.debug("text at current step: {}".format(text_to_split.replace(" ", "<space>")))
                text_before_space = " ".join(text_to_split.split(" ")[:-1])
                logger.debug("before the last space: {}".format(text_before_space.replace(" ", "<space>")))
                if len(text_before_space) > 0:
                    new_tokens = current_tokens.new(self.tokenizer.encode(text_before_space, allowed_special="all")).unsqueeze(0)

            self.tokens.append(new_tokens)

            self.dec_attns = []
            self.kv_cache = {}

            return new_tokens.squeeze(0)
