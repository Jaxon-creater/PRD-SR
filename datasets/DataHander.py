import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch as t
from utils.utils import load_data, load_model, save_model, fix_random_seed_as
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from utils.param import args
import  pickle
import torch, dgl
import sys

if not hasattr(np, '_core'):
    np._core = np.core
if 'numpy._core' not in sys.modules:
    import numpy.core as core
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core._multiarray_umath'] = core._multiarray_umath
    sys.modules['numpy._core.multiarray'] = core.multiarray


def ensure_coo_matrix(x):
    """Convert sparse inputs to scipy coo_matrix robustly across scipy versions."""
    try:
        if sp.issparse(x):
            
            if hasattr(x, 'row') and hasattr(x, 'col'):
                return coo_matrix((np.asarray(x.data), (np.asarray(x.row), np.asarray(x.col))), shape=x.shape)

            if hasattr(x, 'coords'):
                row, col = x.coords
                return coo_matrix((np.asarray(x.data), (np.asarray(row), np.asarray(col))), shape=x.shape)

            if hasattr(x, 'tocoo'):
                y = x.tocoo()
                if hasattr(y, 'row') and hasattr(y, 'col'):
                    return coo_matrix((np.asarray(y.data), (np.asarray(y.row), np.asarray(y.col))), shape=y.shape)
                if hasattr(y, 'coords'):
                    row, col = y.coords
                    return coo_matrix((np.asarray(y.data), (np.asarray(row), np.asarray(col))), shape=y.shape)

            row, col = x.nonzero()
            data = np.asarray(x[row, col]).reshape(-1)
            return coo_matrix((data, (row, col)), shape=x.shape)

        return coo_matrix(x)
    except Exception as e:
        raise TypeError(
            f"Failed to convert object to coo_matrix. type={type(x)}, "
            f"shape={getattr(x, 'shape', None)}, "
            f"format={getattr(x, 'format', None)}, "
            f"has_row={hasattr(x, 'row')}, has_col={hasattr(x, 'col')}, "
            f"has_coords={hasattr(x, 'coords')}, has_tocoo={hasattr(x, 'tocoo')}"
        ) from e

class DataHandler:
    def __init__(self):
        dataset_dirs = {
            'yelp': './datasets/yelp/',
            'ciao': './datasets/ciao/',
            'epinions': './datasets/epinions/',
            'amazon-beauty': './datasets/amazon-beauty/',
            'amazon-health': './datasets/amazon-health/',
        }
        predir = dataset_dirs.get(args.dataset, f'./datasets/{args.dataset}/')
        self.datapath = predir + 'dataset.pkl'
        self.predir = predir
        


    def loadOneFile(self,data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def LoadData(self):
        self.dataset = self.loadOneFile(self.datapath)
        trnMat = self.dataset['train']
        tstMat = self.dataset['test']
        valMat = self.dataset['val']
        trustMat = self.dataset['trust']
        

        trnMat = ensure_coo_matrix(trnMat)
        tstMat = ensure_coo_matrix(tstMat)
        valMat = ensure_coo_matrix(valMat)
        trustMat = ensure_coo_matrix(trustMat)
        self.dataset['trust'] = trustMat
        
        trainset = TrnData(trnMat)
        testset = TstData(tstMat, trnMat)
        valset = TstData(valMat,trnMat)
        self.n_user, self.n_item = self.dataset['userCount'], self.dataset['itemCount']
        args.user, args.item = self.n_user, self.n_item
        

        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        self.uu_graph = dgl.from_scipy(trustMat)
        uimat = trnMat.tocsr()
        self.ui_graph = self.makeBiAdj(uimat,self.n_user,self.n_item)

    def makeBiAdj(self, mat,n_user,n_item):
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=torch.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph


class TrnData(data.Dataset):
    def __init__(self, coomat):
        coomat = ensure_coo_matrix(coomat)
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        coomat = ensure_coo_matrix(coomat)
        self.csrmat = (ensure_coo_matrix(trnMat).tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])



