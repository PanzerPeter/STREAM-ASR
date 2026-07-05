# src/slices/ExtractFeatures/FeatureCollator.py
import torch

from src.shared_kernel.Config_Adapter import get_config
from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch

IGNORE_ID = -100  # cross-entropy ignore_index for padded decoder-output positions


def collate_features(samples: list) -> FeatureBatch:
    mels, token_lists = zip(*samples)
    m = get_config().model

    feat_lengths = torch.tensor([mel.shape[0] for mel in mels], dtype=torch.long)
    tok_lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)

    t_max = int(feat_lengths.max())
    u_max = int(tok_lengths.max())
    n_mels = mels[0].shape[1]

    features = torch.zeros(len(mels), t_max, n_mels, dtype=torch.float32)
    tokens = torch.zeros(len(mels), u_max, dtype=torch.long)

    ud = u_max + 1  # +1 for the SOS/EOS slot
    dec_in_l2r = torch.full((len(mels), ud), m.eos_id, dtype=torch.long)
    dec_in_r2l = torch.full((len(mels), ud), m.eos_id, dtype=torch.long)
    dec_out_l2r = torch.full((len(mels), ud), IGNORE_ID, dtype=torch.long)
    dec_out_r2l = torch.full((len(mels), ud), IGNORE_ID, dtype=torch.long)

    for i, (mel, ids) in enumerate(zip(mels, token_lists)):
        features[i, : mel.shape[0]] = mel
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        rev = ids[::-1]
        n = len(ids)
        dec_in_l2r[i, 0] = m.sos_id
        dec_in_l2r[i, 1 : n + 1] = torch.tensor(ids, dtype=torch.long)
        dec_out_l2r[i, :n] = torch.tensor(ids, dtype=torch.long)
        dec_out_l2r[i, n] = m.eos_id
        dec_in_r2l[i, 0] = m.sos_id
        dec_in_r2l[i, 1 : n + 1] = torch.tensor(rev, dtype=torch.long)
        dec_out_r2l[i, :n] = torch.tensor(rev, dtype=torch.long)
        dec_out_r2l[i, n] = m.eos_id

    return FeatureBatch(
        features,
        feat_lengths,
        tokens,
        tok_lengths,
        dec_in_l2r,
        dec_out_l2r,
        dec_in_r2l,
        dec_out_r2l,
        tok_lengths + 1,
    )
