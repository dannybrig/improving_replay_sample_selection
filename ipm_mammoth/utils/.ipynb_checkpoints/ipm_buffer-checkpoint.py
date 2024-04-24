import sys

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
from . import irlb

def ipm(data, n=1, selected_data=None):
    """
        data: list of data to be selected
        n:    number of samples to be selected
        selected_data: list of already selected data
    """
    if selected_data is None:
        selected_data = []

    pooled_idx = []
    while len(pooled_idx) < n:
        new_idx = ipm_add_sample(selected_data, data, pooled_idx)
        pooled_idx.append(int(new_idx))
    return pooled_idx


def ipm_add_sample(train, pool, pooled_idx):
    candidate_samples = range(0, len(pool))
    set_idx = [int(idx) for idx in pooled_idx]

    # generating the matrix of already selected samples
    A_train = [np.ravel(t) for t in train]
    A_train.extend([np.ravel(pool[i]) for i in set_idx])
    A_s_mat = np.array(A_train).transpose()
    if len(A_s_mat.shape) == 1:
        A_s_mat = A_s_mat.reshape((-1, 1))

    # generating the matrix of data
    A_mat = np.array([np.ravel(t) for t in pool]).transpose()
    if len(A_mat.shape) == 1:
        A_mat = A_mat.reshape((-1, 1))

    # projecting onto the nullspace of the selected data
    if len(A_s_mat) == 0:
        A_proj = A_mat
    else:
        Proj = np.matmul(A_s_mat, np.linalg.pinv(A_s_mat))
        A_proj = A_mat - np.matmul(Proj, A_mat)

    # calculating the first singular vector
    first_eig_vec = irlb.irlb(A_proj, 2)[0][:, 0]

    # calculating the correlations
    correlation = np.zeros(len(pool))
    for m in candidate_samples:

        if m in pooled_idx:
            correlation[m] = 0
            continue

        correlation[m] = np.abs(np.inner(A_mat[:, m], first_eig_vec))
        correlation[m] /= np.linalg.norm(np.squeeze(A_mat[:, m]))

    # finding the best sample
    sorted_idx = np.argsort(correlation)

    return sorted_idx[-1]

class Buffer:
    """
    The memory buffer of rehersal method
    """
    
    def __init__(self, buffer_size, device, cpt, n_tasks = None):
        self.buffer_size = buffer_size
        #self.buffer_portion_size = buffer_size // n_tasks
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.cpt = cpt
        
    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self
    
    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)
    
    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor, 
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.empty(0, device=self.device))
                
    def add_data(self, task_id, examples, features, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param features: tensor containing penultimate features
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if task_id == 0:
            self.init_tensors(examples, labels, logits, task_labels)
            num_to_be_added = self.buffer_size
        else:
            num_to_be_added = self.buffer_size - self.examples.shape[0]
        
        for i in range(self.cpt * task_id, self.cpt * task_id + self.cpt):
            
            if i % 2 == 0:
                samples_to_add = int(np.floor(num_to_be_added / self.cpt))
            else:
                samples_to_add = int(np.ceil(num_to_be_added / self.cpt))
                
            class_idxs = torch.argwhere(labels == i)
            ipm_idxs = ipm(features[class_idxs].detach().cpu().numpy(), n = samples_to_add)
            
            examples_to_add = examples[class_idxs][ipm_idxs].squeeze()
            labels_to_add = labels[class_idxs][ipm_idxs].squeeze()
            logits_to_add = logits[class_idxs][ipm_idxs].squeeze()
            
            self.examples = torch.cat((self.examples, examples_to_add.to(self.device)))
            self.labels = torch.cat((self.labels, labels_to_add.to(self.device)))
            self.logits = torch.cat((self.logits, logits_to_add.to(self.device)))
            
        size = self.examples.shape[0]
        assert self.examples.shape[0] == self.labels.shape[0] == self.logits.shape[0]
        assert size <= self.buffer_size

    
    def delete_data(self, task_id, labels=None, logits=None, task_labels=None):
        """
        Deletes least informative data (according to ipm) for each class.
        """
        
        # Separete data into tensors by splitting on class_ids
        class_ids = torch.unique(self.labels)
        data_partitions = [self.examples[torch.argwhere(self.labels == class_id)].squeeze() for class_id in class_ids]
        label_partitions = [self.labels[torch.argwhere(self.labels == class_id)].squeeze() for class_id in class_ids]
        logit_partitions = [self.logits[torch.argwhere(self.labels == class_id)].squeeze() for class_id in class_ids]
        
        # Delete Data
        for i in range(len(class_ids)):
            
            # To make maximal use of buffer when # samples/class in task is odd
            if i % 2 == 0:
                samples_per_class = int(np.floor(self.buffer_size / (task_id+1) / self.cpt))
            else:
                samples_per_class = int(np.ceil(self.buffer_size / (task_id+1) / self.cpt))
                
            # Edge Cases for when when memory buget runs out and each class only has one sample
            if label_partitions[i].shape == torch.Size([]):
                x = 0
                label_partitions[i] = label_partitions[i].unsqueeze(0)
            elif samples_per_class == 2:
                x = 1
            else:
                # Normal situation when edge cases are not an issue
                x = np.abs(samples_per_class - label_partitions[i].shape[0]) # number of samples to be deleted
                
            # Another edge case scenario to allow for concatenation
            if data_partitions[i].shape == torch.Size([]) or len(data_partitions[i].shape) == 3:
                data_partitions[i] = data_partitions[i].unsqueeze(0)
            if logit_partitions[i].shape == torch.Size([]) or len(logit_partitions[i].shape) == 1:
                logit_partitions[i] = logit_partitions[i].unsqueeze(0)
                
            # When x = 0, this corresponds to nothing to delete which creates empty class arrays for storage
            if x != 0:
                data_partitions[i] = data_partitions[i][:-x]
                label_partitions[i] = label_partitions[i][:-x]
                logit_partitions[i] = logit_partitions[i][:-x]
            else:
                pass
        
        self.examples = torch.cat(data_partitions)
        self.labels = torch.cat(label_partitions)
        self.logits = torch.cat(logit_partitions)
        
        assert self.examples.shape[0] == self.labels.shape[0] == self.logits.shape[0]
    
    def get_data(self, size: int, transform: transforms = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > self.examples.shape[0]:
            size = self.examples.shape[0]

        choice = np.random.choice(self.examples.shape[0], size=size, replace=False)
        
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple
    
    def get_data_by_index(self, indexes, transform: transforms = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple
    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if not hasattr(self, 'examples'):
            return True
        else:
            return False
        
    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple
    
    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0