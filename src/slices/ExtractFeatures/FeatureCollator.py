import torch
from torch.nn.utils.rnn import pad_sequence

from src.slices.ExtractFeatures.FeatureBatch_Response import FeatureBatch


def collate_features(samples: list) -> FeatureBatch:
    mels, token_lists = zip(*samples)
    feat_lengths = torch.tensor([mel.shape[0] for mel in mels], dtype=torch.long)
    tok_lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
    features = pad_sequence(list(mels), batch_first=True).float()
    tokens = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in token_lists], batch_first=True
    )
    return FeatureBatch(features, feat_lengths, tokens, tok_lengths)
