from pathlib import Path

try:
    # HACK: SummaryWriter must be imported at the begining or else it will lead to core dumped
    # This is a known issue: https://github.com/pytorch/pytorch/issues/30651
    from torch.utils.tensorboard.writer import SummaryWriter
except:
    pass


_version_file = Path(__file__).parent.resolve() / "version.txt"
if _version_file.is_file():
    with _version_file.open() as file:
        __version__ = file.read().strip()
else:
    __version__ = "0.0.0+local"
