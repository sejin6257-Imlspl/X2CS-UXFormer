# x2ct_nerf/modules/nerf/model_utils.py
import torch
from x2ct_nerf.modules.nerf import nerf_helpers


def batchify(fn, chunk):
    """`fn`을 chunk 크기로 잘라서 순차 적용 (OOM 방지). axis=1: (B, N, C) 형태를 N축으로 청킹."""
    if chunk is None:
        return fn

    def ret(inputs):
        output_dict = {}
        for i in range(0, inputs.shape[1], chunk):
            for k, v in fn(inputs[:, i:i + chunk]).items():
                output_dict[k] = v if i == 0 else torch.cat([output_dict[k], v], 0)
        return output_dict

    return ret


def run_network(inputs, fn, embed_fn, netchunk=1024 * 64):
    """point feature를 positional encoding 후 network에 통과."""
    inputs_flat = torch.reshape(inputs, [len(inputs), -1, inputs.shape[-1]])

    if hasattr(fn, "encoder") or hasattr(fn, "cond_encoder"):
        coords, features = inputs_flat[..., :3], inputs_flat[..., 3:]
        embedded = torch.cat((embed_fn(coords), features), dim=-1)
    else:
        embedded = embed_fn(inputs_flat)

    outputs_dict = batchify(fn, netchunk)(embedded)
    for k in outputs_dict:
        outputs_dict[k] = torch.reshape(outputs_dict[k], list(inputs.shape[:-1]) + [outputs_dict[k].shape[-1]])
    return outputs_dict


def update_nerf_params(cfg):
    embed_fn, cfg["input_ch"] = nerf_helpers.get_embedder(cfg["multires"])
    cfg["input_ch_views"] = 0
    cfg["output_ch"] = cfg["output_color_ch"]

    def network_query_fn(inputs, network_fn):
        return run_network(inputs, network_fn, embed_fn=embed_fn, netchunk=cfg["netchunk"])

    return cfg, network_query_fn