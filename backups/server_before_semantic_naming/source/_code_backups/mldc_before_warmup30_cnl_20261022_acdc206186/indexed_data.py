"""Stable training-image metadata and the two supervised descriptor objectives."""
import hashlib
import json
import torch
from torch.utils.data import Dataset,DataLoader


class IndexedTrainingDataset(Dataset):
    def __init__(self,dataset):
        self.source=dataset
        self.entries=list(dataset.dataset)
        # Keep the training dataset fingerprint unambiguous under PK sampling.
        paths=[str(row[0]) for row in self.entries]
        if len(set(paths))!=len(paths):
            raise ValueError('Duplicate training image paths make stable indices ambiguous')
        self.fingerprint=hashlib.sha256(json.dumps(
            [(str(row[0]),int(row[1]),int(row[2]),int(row[3])) for row in self.entries],
            ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

    def __len__(self):
        return len(self.source)

    def __getitem__(self,index):
        image,pid,camera,view,_=self.source[index]
        return image,pid,camera,view,index


def indexed_collate(batch):
    images,labels,cameras,views,indices=zip(*batch)
    return (torch.stack(images),*[torch.tensor(values,dtype=torch.long)
                                 for values in (labels,cameras,views,indices)])


def indexed_loader(loader):
    """Preserve the original PK sampler and augmentations; add stable image indices."""
    dataset=IndexedTrainingDataset(loader.dataset)
    result=DataLoader(dataset,batch_sampler=loader.batch_sampler,num_workers=loader.num_workers,
        collate_fn=indexed_collate,pin_memory=loader.pin_memory,worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,persistent_workers=loader.persistent_workers)
    return result,dataset

