import importlib


def _safe_star_import(module_name: str):
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return

    names = getattr(mod, "__all__", None)
    if names is None:
        names = [k for k in mod.__dict__.keys() if not k.startswith("_")]

    g = globals()
    for name in names:
        if hasattr(mod, name):
            g[name] = getattr(mod, name)


_safe_star_import("s3prl.downstream.timit_phone.hubconf")
_safe_star_import("s3prl.upstream.apc.hubconf")
_safe_star_import("s3prl.upstream.ast.hubconf")
_safe_star_import("s3prl.upstream.audio_albert.hubconf")
_safe_star_import("s3prl.upstream.baseline.hubconf")
_safe_star_import("s3prl.upstream.byol_a.hubconf")
_safe_star_import("s3prl.upstream.byol_s.hubconf")
_safe_star_import("s3prl.upstream.cpc.hubconf")
_safe_star_import("s3prl.upstream.data2vec.hubconf")
_safe_star_import("s3prl.upstream.decoar2.hubconf")
_safe_star_import("s3prl.upstream.decoar.hubconf")
_safe_star_import("s3prl.upstream.decoar_layers.hubconf")
_safe_star_import("s3prl.upstream.distiller.hubconf")
_safe_star_import("s3prl.upstream.dtmae.hubconf")
_safe_star_import("s3prl.upstream.espnet_hubert.hubconf")
_safe_star_import("s3prl.upstream.example.hubconf")
_safe_star_import("s3prl.upstream.hf_hubert.hubconf")
_safe_star_import("s3prl.upstream.hf_wav2vec2.hubconf")
_safe_star_import("s3prl.upstream.hubert.hubconf")
_safe_star_import("s3prl.upstream.lighthubert.hubconf")
_safe_star_import("s3prl.upstream.log_stft.hubconf")
_safe_star_import("s3prl.upstream.mae_ast.hubconf")
_safe_star_import("s3prl.upstream.mockingjay.hubconf")
_safe_star_import("s3prl.upstream.mos_prediction.hubconf")
_safe_star_import("s3prl.upstream.multires_hubert.hubconf")
_safe_star_import("s3prl.upstream.npc.hubconf")
_safe_star_import("s3prl.upstream.pase.hubconf")
_safe_star_import("s3prl.upstream.passt.hubconf")
_safe_star_import("s3prl.upstream.roberta.hubconf")
_safe_star_import("s3prl.upstream.ssast.hubconf")
_safe_star_import("s3prl.upstream.tera.hubconf")
_safe_star_import("s3prl.upstream.unispeech_sat.hubconf")
_safe_star_import("s3prl.upstream.vggish.hubconf")
_safe_star_import("s3prl.upstream.vq_apc.hubconf")
_safe_star_import("s3prl.upstream.vq_wav2vec.hubconf")
_safe_star_import("s3prl.upstream.wav2vec2.hubconf")
_safe_star_import("s3prl.upstream.wav2vec.hubconf")
_safe_star_import("s3prl.upstream.wavlm.hubconf")


def options(only_registered_ckpt: bool = False):
    all_options = []
    for name, value in globals().items():
        torch_hubconf_policy = not name.startswith("_") and callable(value)
        if torch_hubconf_policy and name != "options":
            if only_registered_ckpt and (
                name.endswith("_local")
                or name.endswith("_url")
                or name.endswith("_gdriveid")
                or name.endswith("_custom")
            ):
                continue
            all_options.append(name)

    return all_options
