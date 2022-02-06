from . import transforms as T


def build_transforms(cfg, is_train=True):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.ToBGR(),
            T.To255()
        ]
    )
    return transform
