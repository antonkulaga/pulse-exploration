"""The three open PPG encoders, loaded from their upstream checkouts at pinned commits.

None of the upstream code is copied here: AnyPPG's repository has no licence, so it may be run but not
redistributed. ``FM_HOME`` (default ``~/fm-baselines``) holds

* ``upstream/<repo>``: ``git clone`` of each repository, checked against :data:`COMMITS`;
* ``weights/``: the downloaded checkpoints, checked against :data:`SHA256`.

Only the network-definition file of each repository is imported (by path, not through ``sys.path``), and
every checkpoint is read with ``torch.load(weights_only=True)``, so no pickled code runs.

| name | upstream | input it was trained on | what we feed it |
|---|---|---|---|
| ``pulseppg`` | maxxu05/pulseppg, MIT | wrist PPG, 50 Hz, 4 min, InstanceNorm inside | 30 s at 50 Hz |
| ``papagei_s`` | Nokia-Bell-Labs/papagei-foundation-model, BSD-3-Clear code, weights unlicensed | finger PPG, 125 Hz, 10 s | three 10 s segments at 125 Hz, z-scored, embeddings averaged |
| ``anyppg`` | PKUDigitalHealth/AnyPPG, no licence | finger PPG, 125 Hz, 10 s, z-scored | as PaPaGei |
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import zipfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

COMMITS = {
    "pulseppg": "716eaf9cf966e8f76436f2263872ef38b1f90166",
    "papagei-foundation-model": "0c537dad4d2850e15b724260de820dd68d77f0b0",
    "AnyPPG": "661b877aa96eac3bba320a462cb2a3bfea991103",
}
SHA256 = {
    "pulseppg_model_weights.zip": "5685093114036e9ceb720b98342d68188c8786ed4e48466aa608a325cd890918",
    "papagei_s.pt": "79d68671e51bdc951fee853e5a8241099e38148ff26ed888b843849332e0b988",
    "anyppg_ckpt.pth": "99b9bb0a3c2b83a1f5d8ca2963fbd25329b6530e8d337de8825722fc6fd5f4fa",
}


def fm_home() -> Path:
    return Path(os.environ.get("FM_HOME", str(Path.home() / "fm-baselines")))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checked(path: Path) -> bytes:
    data = path.read_bytes()
    want = SHA256.get(path.name)
    if want and _sha256(data) != want:
        raise RuntimeError(f"{path} has SHA-256 {_sha256(data)}, expected {want}")
    return data


def _git_head(root: Path) -> str:
    """The checked-out commit, read from ``.git`` directly (compute nodes have no git)."""
    git = root / ".git"
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    if (git / ref).is_file():
        return (git / ref).read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve {ref} in {git}")


def _upstream(repo: str) -> Path:
    root = fm_home() / "upstream" / repo
    head = _git_head(root)
    if head != COMMITS[repo]:
        raise RuntimeError(f"{root} is at {head}, expected {COMMITS[repo]}")
    return root


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_dict(obj, marker: str) -> dict[str, torch.Tensor]:
    """The state dict inside a checkpoint whose keys contain ``marker``, with any prefix before it removed."""
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        hits = [k for k in obj if marker in k]
        if hits:
            prefix = hits[0][:hits[0].index(marker)]
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in obj.items()}
    if isinstance(obj, dict):
        for v in obj.values():
            try:
                return _state_dict(v, marker)
            except KeyError:
                continue
    raise KeyError(f"no state dict with {marker!r}")


@dataclass
class Encoder:
    name: str
    fs: float              # rate the network reads
    segment_s: float       # seconds per network input
    zscore: bool           # z-score each segment before the network
    net: torch.nn.Module
    dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, tuple):  # PaPaGei-S: (embedding, moe1, moe2, pooled); the README uses [0]
            out = out[0]
        return out

    def segments(self, x: np.ndarray, fs_in: float) -> np.ndarray:
        """``(n, L)`` windows at ``fs_in`` -> ``(n, k, segment)`` network inputs at :attr:`fs`."""
        ratio = Fraction(self.fs / fs_in).limit_denominator(1000)
        if ratio != 1:
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
        seg = int(round(self.segment_s * self.fs))
        k = x.shape[1] // seg
        if k == 0:
            raise ValueError(f"window of {x.shape[1] / self.fs:.1f} s is shorter than {self.segment_s} s")
        x = x[:, :k * seg].reshape(len(x), k, seg)
        if self.zscore:
            x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        return x.astype(np.float32)

    @torch.inference_mode()
    def embed(self, x: np.ndarray, fs_in: float, device: str, batch: int = 512) -> np.ndarray:
        """One embedding per window: the mean over its segments."""
        out = np.empty((len(x), self.dim), np.float32)
        for i in range(0, len(x), batch):
            s = self.segments(x[i:i + batch], fs_in)
            n, k, seg = s.shape
            t = torch.from_numpy(s.reshape(n * k, 1, seg)).to(device)
            out[i:i + n] = self.forward(t).float().reshape(n, k, -1).mean(1).cpu().numpy()
        return out


def load_pulseppg() -> Encoder:
    root = _upstream("pulseppg")
    net_mod = _import(root / "pulseppg/nets/ResNet1D/ResNet1D_Net.py", "pulseppg_resnet1d")
    # PulsePPG_expconfigs.py: in_channels 1, base_filters 128, kernel 11, stride 2, groups 1, 12 blocks, max pool
    net = net_mod.Net(in_channels=1, base_filters=128, kernel_size=11, stride=2, groups=1, n_block=12,
                      finalpool="max")
    with zipfile.ZipFile(io.BytesIO(_checked(fm_home() / "weights/pulseppg_model_weights.zip"))) as z:
        blob = z.read("pulseppg/experiments/out/pulseppg/checkpoint_best.pkl")
    ckpt = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    net.load_state_dict(_state_dict(ckpt, "first_block_conv"), strict=True)
    return Encoder("pulseppg", 50.0, 30.0, False, net.eval(), 512)


def load_papagei_s() -> Encoder:
    root = _upstream("papagei-foundation-model")
    net_mod = _import(root / "models/resnet.py", "papagei_resnet")
    # README: base_filters 32, kernel 3, stride 2, groups 1, 18 blocks, 512-d, 3 experts
    net = net_mod.ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2, groups=1, n_block=18,
                              n_classes=512, n_experts=3)
    ckpt = torch.load(io.BytesIO(_checked(fm_home() / "weights/papagei_s.pt")), map_location="cpu",
                      weights_only=True)
    net.load_state_dict(_state_dict(ckpt, "first_block_conv"), strict=True)
    return Encoder("papagei_s", 125.0, 10.0, True, net.eval(), 512)


def load_anyppg() -> Encoder:
    root = _upstream("AnyPPG")
    net_mod = _import(root / "load_anyppg/resnet1d.py", "anyppg_resnet1d")
    # load_anyppg/demo_load_anyppg.py
    net = net_mod.Net1D(in_channels=1, base_filters=64, ratio=1.0, filter_list=[64, 160, 160, 400, 400, 512],
                        m_blocks_list=[2, 2, 2, 3, 3, 1], kernel_size=3, stride=2, groups_width=16,
                        use_bn=True, use_do=True, verbose=False)
    ckpt = torch.load(io.BytesIO(_checked(root / "load_anyppg/anyppg_ckpt.pth")), map_location="cpu",
                      weights_only=True)
    net.load_state_dict(_state_dict(ckpt, "first_conv"), strict=True)
    return Encoder("anyppg", 125.0, 10.0, True, net.eval(), 512)


LOADERS = {"pulseppg": load_pulseppg, "papagei_s": load_papagei_s, "anyppg": load_anyppg}
