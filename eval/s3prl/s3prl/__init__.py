from pathlib import Path

import torchaudio

try:
    # HACK: SummaryWriter must be imported at the begining or else it will lead to core dumped
    # This is a known issue: https://github.com/pytorch/pytorch/issues/30651
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    pass


if not hasattr(torchaudio, "set_audio_backend"):
    def _set_audio_backend_compat(*args, **kwargs):
        return None

    torchaudio.set_audio_backend = _set_audio_backend_compat


_version_file = Path(__file__).parent.resolve() / "version.txt"
if _version_file.is_file():
    with _version_file.open() as file:
        __version__ = file.read().strip()
else:
    __version__ = "0.0.0+local"
