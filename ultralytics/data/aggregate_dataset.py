from torch.utils.data import Dataset

from ultralytics.data import YOLODataset
from ultralytics.utils import colorstr


class DatasetAggregator(YOLODataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(num_of_samples for dataset, num_of_samples,_ in self.datasets)

    def __getitem__(self, idx):
        for dataset, max_samples_per_ds, head_name in self.datasets:
            if idx < max_samples_per_ds:
                sample = dataset[idx]
                sample['head_name'] = head_name

                return sample
            idx -= max_samples_per_ds
        raise IndexError(f"Index {idx} out of range for aggregated dataset")


def build_aggregate_dataset(cfg, batch, datas, mode="train", rect=False, stride=32):
    datasets = []
    for data in datas:
        img_path = data["train"] if mode == "train" else data["val"]
        dataset = YOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
        # TODO: move to config
        samples_per_dataset = min(len(dataset), 100000000)

        # TODO: how to do the randomize thing?
        datasets.append((dataset, samples_per_dataset, data["head_name"]))
    return DatasetAggregator(datasets)
