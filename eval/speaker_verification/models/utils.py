import sys
from pathlib import Path

import torch
from packaging import version
import torch.nn.functional as F
from omegaconf import OmegaConf
from s3prl.upstream.interfaces import UpstreamBase
from torch.nn.utils.rnn import pad_sequence

_FAIRSEQ_ROOT = Path(__file__).resolve().parents[2] / "fairseq"
_FAIRSEQ_PKG = _FAIRSEQ_ROOT / "fairseq" / "__init__.py"
if _FAIRSEQ_PKG.is_file():
    fairseq_root_str = str(_FAIRSEQ_ROOT)
    if fairseq_root_str not in sys.path:
        sys.path.insert(0, fairseq_root_str)

import fairseq

try:
    from fairseq import tasks
    from fairseq.checkpoint_utils import load_checkpoint_to_cpu
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
except Exception as e:
    raise ImportError(
        "Failed to import fairseq runtime modules. "
        "Ensure eval/fairseq is populated with your local project version "
        "(including any custom patches), then rerun."
    ) from e

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    # state = load_checkpoint_to_cpu(filepath)
    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        assert version.parse(fairseq.__version__) > version.parse(
            "0.10.2"
        ), "Please install the fairseq master branch."

        model, cfg, task = load_model(ckpt)
        self.model = model
        self.task = task

        if len(self.hooks) == 0:
            module_name = "self.model.encoder.layers"
            for module_id in range(len(eval(module_name))):
                self.add_hook(
                    f"{module_name}[{module_id}]",
                    lambda input, output: input[0].transpose(0, 1),
                )
            self.add_hook("self.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )
        return {
            "default": features,
        }
