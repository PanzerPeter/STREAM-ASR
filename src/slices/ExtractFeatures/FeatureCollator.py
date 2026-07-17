# src/slices/ExtractFeatures/FeatureCollator.py
import torch

from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch


def collate_features(samples: list) -> FeatureBatch:
    mels, token_lists = zip(*samples)
    feat_lengths = torch.tensor([mel.shape[0] for mel in mels], dtype=torch.long)
    tok_lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
    t_max = int(feat_lengths.max())
    u_max = int(tok_lengths.max())
    n_mels = mels[0].shape[1]
    features = torch.zeros(len(mels), t_max, n_mels, dtype=torch.float32)
    tokens = torch.zeros(len(mels), u_max, dtype=torch.long)
    for i, (mel, ids) in enumerate(zip(mels, token_lists)):
        features[i, : mel.shape[0]] = mel
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return FeatureBatch(features, feat_lengths, tokens, tok_lengths)
