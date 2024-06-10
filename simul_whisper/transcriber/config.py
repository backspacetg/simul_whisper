from dataclasses import dataclass, field

@dataclass
class AlignAttConfig:
    model_path: str
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata = {"help": "in second"})
    language: str = field(default="zh")
    frame_threshold: int = 4
    nonspeech_prob: float = 1.0
    rewind_threshold: int = 200
    buffer_len: float = 30.0
    min_seg_len: float = 1.0
    if_ckpt_path: str = ""