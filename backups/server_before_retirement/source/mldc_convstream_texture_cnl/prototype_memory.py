"""Training-only identity prototypes, updated only after successful SGD steps."""
import torch
from torch import nn
from torch.nn import functional as F


class PrototypeMemory(nn.Module):
    def __init__(self, image_labels, num_classes, dim=768, momentum=.5, min_images=2,
                 update_mode='adaptive', amu_temperature=.05):
        super().__init__()
        labels = torch.as_tensor(image_labels, dtype=torch.long)
        if labels.ndim != 1 or not len(labels) or labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError('Prototype memory requires official relabeled training identities')
        if not 0 <= momentum < 1 or min_images < 1:
            raise ValueError('Invalid prototype EMA or unique-image eligibility')
        if update_mode not in ['adaptive','fixed'] or amu_temperature <= 0:
            raise ValueError('AMU requires a positive temperature and a known update mode')
        self.register_buffer('image_labels', labels)
        self.register_buffer('seen_images', torch.zeros(len(labels), dtype=torch.bool))
        self.register_buffer('features', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('image_counts', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('updates', torch.zeros(num_classes, dtype=torch.long))
        self.momentum = float(momentum)  # Weight on the OLD prototype.
        self.min_images = int(min_images)
        self.update_mode = update_mode
        self.amu_temperature = float(amu_temperature)

    def validate(self, features, labels, indices):
        if features.ndim != 2 or features.shape != (len(labels), self.features.shape[1]):
            raise ValueError('Prototype embedding shape does not match identity memory')
        if labels.shape != indices.shape or labels.ndim != 1 or not len(labels):
            raise ValueError('Training identities and image indices must align')
        if indices.min() < 0 or indices.max() >= len(self.image_labels):
            raise ValueError('Prototype image index outside official training data')
        if not torch.equal(labels, self.image_labels[indices]):
            raise ValueError('Prototype identity/index mismatch')

    @torch.no_grad()
    def image_means(self, features, labels, indices):
        self.validate(features, labels, indices)
        unique, inverse = indices.unique(sorted=True, return_inverse=True)
        means = features.new_zeros((len(unique), features.shape[1]), dtype=torch.float32)
        means.index_add_(0, inverse, F.normalize(features.detach().float(), dim=1))
        return unique, self.image_labels[unique], F.normalize(means, dim=1)

    @torch.no_grad()
    def lookup(self, labels):
        return (self.features[labels].detach().clone(),
                (self.valid[labels] & (self.image_counts[labels] >= self.min_images)).clone())

    @torch.no_grad()
    def classification_keys(self, features, labels, indices):
        """Bootstrap missing classes from detached current image means, without writes."""
        _, image_labels, values = self.image_means(features, labels, indices)
        classes, inverse = image_labels.unique(sorted=True, return_inverse=True)
        current = values.new_zeros((len(classes), values.shape[1]))
        current.index_add_(0, inverse, values)
        current = F.normalize(current, dim=1)
        bank = self.features.detach().clone()
        mask = self.valid.clone()
        fresh = ~mask[classes]
        bank[classes[fresh]] = current[fresh]
        mask[classes] = True
        ids = mask.nonzero(as_tuple=True)[0]
        columns = torch.full_like(self.image_counts, -1)
        columns[ids] = torch.arange(len(ids), device=ids.device)
        return bank[ids], columns[labels]

    @torch.no_grad()
    def update(self, features, labels, indices):
        unique, image_labels, values = self.image_means(features, labels, indices)
        # Eq. (15): select the negative by prototype-to-prototype similarity.
        # Eq. (16): use that negative and EACH refined image embedding to set mu.
        # Coalesce augmentations of the same image; process distinct images in
        # stable image-index order. Subsequent images read the updated bank.
        rates=[];bootstrap=0
        for identity, value in zip(image_labels.tolist(), values):
            if not bool(self.valid[identity]):
                self.features[identity] = value
                self.valid[identity] = True
                bootstrap += 1
            else:
                old = self.features[identity].clone()
                candidates = self.valid.clone()
                candidates[identity] = False
                if self.update_mode == 'adaptive' and bool(candidates.any()):
                    ids = candidates.nonzero(as_tuple=True)[0]
                    negative = self.features[ids[(self.features[ids] @ old).argmax()]].clone()
                    margin = (old @ value - negative @ value) / self.amu_temperature
                    mu = torch.sigmoid(margin.float())
                else:
                    mu = value.new_tensor(1-self.momentum)
                self.features[identity] = F.normalize((1-mu)*old + mu*value,dim=0)
                rates.append(float(mu))
            self.updates[identity] += 1
        newly_seen = ~self.seen_images[unique]
        self.image_counts.index_add_(0, image_labels[newly_seen], torch.ones_like(image_labels[newly_seen]))
        self.seen_images[unique] = True
        return dict(amu_mu_sum=sum(rates),amu_mu_min=min(rates) if rates else None,
                    amu_mu_max=max(rates) if rates else None,amu_samples=len(rates),
                    bootstrap_samples=bootstrap)
