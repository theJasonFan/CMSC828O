#import click
import numpy as np 
from scipy.special import expit as sigmoid

def gen_sbm(n_per_clust, k_clusters):
    n_nodes = n_per_clust * k_clusters
    nodes = np.arange(n_nodes)
    node_labels = np.sort(nodes % k_clusters)
    #print(nodes, node_labels)

    # sample a P- matrix
    clust_ps = np.random.randn(k_clusters, k_clusters) - 1
    clust_ps = clust_ps + clust_ps.T
    clust_ps = sigmoid(clust_ps)
    clust_ps[np.eye(k_clusters, dtype=bool)] = 0.9

    #print(clust_ps)

    A = np.random.uniform(0, 1, (n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            z = A[i,j]
            p = clust_ps[node_labels[i], node_labels[j]]
            A[i, j] = z < p
            A[j, i] = z < p 
    
    A[np.eye(len(A), dtype=bool)] = 0
    
        
    return A, node_labels, clust_ps
            

# @click.command()
# @click.option('-n', '--n_per_clust', type=int)
# @click.option('-k', '--k_clusters', type=int)
# @click.option('-oa', '--out_adj')
# @click.option('-on', '--out_nodes')
# @click.option('-oc', '--out_ps')
# def cli(n_per_clust, k_clusters, out_adj, out_nodes, out_ps):
#     gen_sbm(n_per_clust, k_clusters)




# if __name__ == "__main__":
#     cli()

class SBM(object):
    def __init__(self, A, k, tol=1e-6, max_iters=20, max_e_iters=100):
        assert(np.allclose(np.diag(A), 0))
        self.tol = tol
        self.A = A
        self.k = k
        self.n = len(A)
        self.max_iters = max_iters
        self.max_e_iters = max_e_iters
        
        # cluster linkage probabality
        self.theta = np.random.uniform(high=1, size=(self.k, self.k))
        #self.theta += self.theta.T

        # cluster prior. (probability any node is assigned a cluster)
        self.alpha = np.random.uniform(size=self.k)
        self.alpha = self.alpha / np.sum(self.alpha)  

        # per node cluster probabality
        self.gamma = np.random.uniform(size=(self.n, self.k))
        self.gamma = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)

        # Zs
        self.Z = np.random.randint(low=0, high=self.k, size=self.n)

        assert(np.allclose(self.alpha.sum(), 1))
        assert(np.allclose(np.sum(self.gamma, axis=1), 1))
        assert(all(self.Z < k))

    def fit(self, verbose=True):
        self.alpha = self._M_alpha(self.gamma, self.n)
        self.theta = self._M_theta(self.A, self.gamma)
        for i in range(self.max_iters):
            if verbose: print(i)
            self.gamma = self._E(verbose)
            self.alpha = self._M_alpha(self.gamma, self.n)
            self.theta = self._M_theta(self.A, self.gamma)
    
        return self
    
    def _E(self, verbose=True):
        rel_err = 1
        tau_old = self.gamma.copy()
        for it in range(self.max_e_iters):
            tau_new = tau_old.copy()
            for i in range(self.n):
                for q in range(self.k):
                    tau_new[i,q] = self.iter_fixed_point(i, q, tau_old)
            
            #print(tau_new)
            tau_new = tau_new / np.sum(tau_new, axis=1, keepdims=True)
            rel_err = np.linalg.norm(tau_new - tau_old)
            if verbose and (it + 1) % 10 == 0: print("\t[%d] E rel_err: %f" %(it,rel_err))
            if rel_err < self.tol: break
            tau_old = tau_new
        if verbose: print("\t[%d] E rel_err: %f" %(it,rel_err))
        return tau_old


    def iter_fixed_point(self, i, q, tau_old):
        prod = self.alpha[q]
        for j in range(self.n):
            if i == j: continue
            for l in range(self.k):
                y_ij = self.A[i, j]
                theta_ql = self.theta[q,l]
                theta_lq = self.theta[l,q]
                if y_ij == 1:
                    prod *= np.power(theta_ql*theta_lq, tau_old[j, l])
                else:
                    prod *= np.power((1 - theta_ql)*(1-theta_lq), tau_old[j, l])
                assert(~np.isnan(prod))

        return prod
    
    @staticmethod
    def _M_alpha(gamma, n):
        alpha = np.sum(gamma, axis=0) / n
        assert(np.allclose(np.sum(alpha), 1))
        assert(alpha.shape == (gamma.shape[1],))
        return alpha
    
    @staticmethod
    def _M_theta(Y, gamma):
        k = gamma.shape[1]
        n = len(Y)

        theta = np.zeros((k, k))
        for q in range(k):
            for r in range(q, k):
                gamma_q = gamma[:, q]
                gamma_r = gamma[:, r]

                sum_qr = np.sum((np.ones((n,n)) - np.eye(n)) * np.outer(gamma_q, gamma_r))
                #sum_qr = np.sum(np.outer(gamma_q, gamma_r))
                sum_yqr = np.sum(Y * np.outer(gamma_q, gamma_r))
                # print(sum_qr, sum_yqr)
                theta[q,r] = sum_yqr / sum_qr
                theta[r,q] = sum_yqr / sum_qr
        return theta


    def L(self):
        return 0
