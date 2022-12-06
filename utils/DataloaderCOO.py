import collections
import torch


TorchCSR = collections.namedtuple("TrochCSR", "data indices indptr shape")

def load_csr_data_to_gpu(train_inputs,device = torch.device("cuda:0")):
    """Move a scipy csr sparse matrix to the gpu as a TorchCSR object
    This try to manage memory efficiently by creating the tensors and moving them to the gpu one by one
    """
    th_data = torch.from_numpy(train_inputs.data).to(device)
    th_indices = torch.from_numpy(train_inputs.indices).to(device)
    th_indptr = torch.from_numpy(train_inputs.indptr).to(device)
    th_shape = train_inputs.shape
    return TorchCSR(th_data, th_indices, th_indptr, th_shape)

def load_csr_data_to_cpu(train_inputs):
    """Move a scipy csr sparse matrix to the gpu as a TorchCSR object
    This try to manage memory efficiently by creating the tensors and moving them to the gpu one by one
    """
    th_data = torch.from_numpy(train_inputs.data).to("cpu")
    th_indices = torch.from_numpy(train_inputs.indices).to("cpu")
    th_indptr = torch.from_numpy(train_inputs.indptr).to("cpu")
    th_shape = train_inputs.shape
    return TorchCSR(th_data, th_indices, th_indptr, th_shape)


def make_coo_batch(torch_csr, indx):
    """Make a coo torch tensor from a TorchCSR object by taking the rows indicated by the indx tensor
    """
    th_data, th_indices, th_indptr, th_shape = torch_csr
    indx = indx.to(dtype = torch.long)
    start_pts = th_indptr[indx]
    end_pts = th_indptr[indx+1]
    coo_data = torch.cat([th_data[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_col = torch.cat([th_indices[start_pts[i]: end_pts[i]] for i in range(len(start_pts))], dim=0)
    coo_row = torch.repeat_interleave(torch.arange(indx.shape[0]), th_indptr[indx+1] - th_indptr[indx])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [indx.shape[0], th_shape[1]])
    return coo_batch

def make_coo_batch_slice(torch_csr, start, end):
    """Make a coo torch tensor from a TorchCSR object by taking the rows within the (start, end) slice
    """
    th_data, th_indices, th_indptr, th_shape = torch_csr
    if end > th_shape[0]:
        end = th_shape[0]
    start_pts = th_indptr[start]
    end_pts = th_indptr[end]
    coo_data = th_data[start_pts: end_pts]
    coo_col = th_indices[start_pts: end_pts]
    coo_row = torch.repeat_interleave(torch.arange(end-start, device="cpu"), th_indptr[start+1:end+1] - th_indptr[start:end])
    coo_batch = torch.sparse_coo_tensor(torch.vstack([coo_row, coo_col]), coo_data, [end-start, th_shape[1]])
    return coo_batch

class DataLoaderCOO:
    """Torch compatible DataLoader. Works with in-device TorchCSR tensors.
    Args:
         - train_inputs, train_targets: TorchCSR tensors
         - train_idx: tensor containing the indices of the rows of train_inputs and train_targets that should be used
         - batch_size, shuffle, drop_last: as in torch.utils.data.DataLoader
    """
    def __init__(self, train_inputs, train_targets, train_idx=None, 
                 *,
                batch_size=512, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        
        self.train_idx = train_idx
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else train_inputs.shape[0]
        self.nb_batches = self.nb_examples//batch_size
        if not drop_last and not self.nb_examples%batch_size==0:
            self.nb_batches +=1
        
    def __iter__(self,device = torch.device("cuda:0")):
        if self.shuffle:
            shuffled_idx = torch.randperm(self.nb_examples, device=device)
            if self.train_idx is not None:
                idx_array = self.train_idx[shuffled_idx]
            else:
                idx_array = shuffled_idx
        else:
            if self.train_idx is not None:
                idx_array = self.train_idx
            else:
                idx_array = None
            
        for i in range(self.nb_batches):
            slc = slice(i*self.batch_size, (i+1)*self.batch_size)
            if idx_array is None:
                inp_batch = make_coo_batch_slice(self.train_inputs, i*self.batch_size, (i+1)*self.batch_size)
                if self.train_targets is None:
                    tgt_batch = None
                else:
                    tgt_batch = make_coo_batch_slice(self.train_targets, i*self.batch_size, (i+1)*self.batch_size)
            else:
                idx_batch = idx_array[slc]
                inp_batch = make_coo_batch(self.train_inputs, idx_batch)
                if self.train_targets is None:
                    tgt_batch = None
                else:
                    tgt_batch = make_coo_batch(self.train_targets, idx_batch)
            yield inp_batch, tgt_batch
            
            
    def __len__(self):
        return self.nb_batches