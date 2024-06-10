import os
import sys
import torch

sys.path.append(os.path.dirname(__file__))
from simul_whisper.transcriber.config import AlignAttConfig
from simul_whisper.transcriber.segment_loader import SegmentWrapper
from simul_whisper.transcriber.simul_whisper import PaddedAlignAttWhisper, DEC_PAD

model_path = "path-to-the-whisper-checkpoint"
if_ckpt_path = "cif_models/small.pt" # align with the whisper model. e.g., using small.pt for whisper small

segment_length = 1.0 # chunk length, in seconds
frame_threshold = 12 # threshold for the attention-guided decoding, in frames
buffer_len = 20 # the lengths for the context buffer, in seconds
min_seg_len = 0.0 # transcibe only when the context buffer is larger than this threshold. Useful when the segment_length is small
language = "en"

audio_path = "path-to-the-audio-file"

if __name__ == "__main__":

    cfg = AlignAttConfig(
        model_path=model_path, 
        segment_length=segment_length,
        frame_threshold=frame_threshold,
        language=language,
        buffer_len=buffer_len, 
        min_seg_len=min_seg_len,
        if_ckpt_path=if_ckpt_path,
    )

    model = PaddedAlignAttWhisper(cfg)
    segmented_audio = SegmentWrapper(audio_path=audio_path, segment_length=segment_length)

    hyp_list = []
    for seg_id, (seg, is_last) in enumerate(segmented_audio):
        new_toks = model.infer(seg, is_last)
        hyp_list.append(new_toks)
        hyp = torch.cat(hyp_list, dim=0)
        hyp = hyp[hyp < DEC_PAD]
        hyp = model.tokenizer.decode(hyp)
        print(hyp)

    model.refresh_segment(complete=True) # refresh the buffer when an utterance is decoded
