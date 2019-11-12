#import click
import numpy as np 

def gen_sbm(n_per_clust, k_clusters):
    from scipy.special import expit as sigmoid
    n_nodes = n_per_clust * k_clusters
    nodes = np.arange(n_nodes)
    node_labels = np.sort(nodes % k_clusters)
    #print(nodes, node_labels)

    # sample a P- matrix
    clust_ps = np.random.randn(k_clusters, k_clusters) - 1
    clust_ps = clust_ps + clust_ps.T
    clust_ps = sigmoid(clust_ps)
    clust_ps[np.eye(k_clusters, dtype=bool)] = 0.9

    A = np.random.uniform(0, 1, (n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            z = A[i,j]
            p = clust_ps[node_labels[i], node_labels[j]]
            A[i, j] = z < p
            A[j, i] = z < p 
    
    A[np.eye(len(A), dtype=bool)] = 0
    
        
    return A, node_labels, clust_ps

class SBM(object):
    def __init__(self, A, k, tol=1e-6, max_iters=20, max_e_iters=100, init='spectral'):
        assert(np.allclose(np.diag(A), 0))
        assert(np.allclose(A, A.T))
        self.tol = tol
        self.A = A
        self.k = k
        self.n = len(A)
        self.max_iters = max_iters
        self.max_e_iters = max_e_iters

        # Set initial Z assignment
        if init is 'random':
            self.random_init()
        elif init == 'spectral':
            self.spectral_init()
        else:
            raise NotImplementedError
        self.init_from_Z()

        assert(np.allclose(self.alpha.sum(), 1))
        assert(np.allclose(np.sum(self.gamma, axis=1), 1))
        assert(all(self.Z < k))
    
    def spectral_init(self):
        from sklearn.cluster import SpectralClustering
        spectral_clust = SpectralClustering(n_clusters=self.k, affinity='precomputed')
        spectral_clust.fit(self.A)
        self.Z = spectral_clust.fit_predict(self.A)
    
    def random_init(self):
        self.Z = np.random.randint(low=0, high=self.k, size=self.n)
    
    def init_from_Z(self, init_noise=1e-4):
        # Initialize parameters from cluster assignment labels.
        self.gamma = np.zeros((self.n, self.k))
        self.gamma[range(self.n), self.Z] = 1
        self.gamma = self.gamma + init_noise
        self.gamma = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)

        self.alpha = self._M_alpha(self.gamma)
        self.theta = self._M_theta(self.A, self.gamma)

    def fit(self, verbose=False, verbose_E=False, warn=True):
        NLL_old  = self.NLL()
        for i in range(self.max_iters):
            self.gamma = self._E_fast(verbose_E)
            self.alpha = self._M_alpha(self.gamma)
            self.theta = self._M_theta(self.A, self.gamma)
            self.Z = self.MAP_Z()
            NLL_new = self.NLL()

            if verbose: print('[ITER]', i, 'NLL=%f' % self.NLL())

            if NLL_new <= NLL_old and np.abs(NLL_old - NLL_new) < self.tol:
                if verbose: print("\tConverged...")
                break
            NLL_old = NLL_new
        self.iters_ran = i + 1
        return self

    def _E_fast(self, verbose=True):
        # Matrix multiplication for the fixed point is necessary for 
        # a sane, reasonably fast model. 
        # https://github.com/cran/blockmodels/blob/eb543aba11cb0ece159fa7dc01d1eb8bb9d32ed2/src/models/bernoulli.h#L89
        rel_err = 1
        tau_old = self.gamma.copy()
        for it in range(self.max_e_iters):

            tau_new = tau_old.copy()
            log_1 = np.linalg.multi_dot([self.A, tau_new, np.log(self.theta) + np.log(self.theta.T)])
            Y_not = 1 - self.A - np.eye(len(self.A))
            log_2 = np.linalg.multi_dot([Y_not, tau_new, np.log(1 - self.theta) + np.log(1 - self.theta.T)])
            tau_new_log = np.log(self.alpha) + log_1 + log_2
            
            # Clipping and manipulation to avoid overflow/underflow
            tau_new_log = tau_new_log - np.max(tau_new_log, axis=1, keepdims=True)
            tau_new = np.exp(tau_new_log) + 1e-9

            if np.any(np.isnan(tau_new)): 
                assert(False)
            tau_new = tau_new / np.sum(tau_new, axis=1, keepdims=True)
            rel_err = np.linalg.norm(tau_new - tau_old)
            if verbose and (it + 1) % 10 == 0: print("\t[%d] E rel_err: %f" %(it,rel_err))

            tau_old = tau_new
            if rel_err < self.tol: 
                break
        if verbose: print("\t[%d] E rel_err: %f" %(it,rel_err))
        return tau_old

    @staticmethod
    def _M_alpha(gamma):
        alpha = np.sum(gamma, axis=0) / len(gamma)
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
    
    def NLL(self):
        Z = self.Z
        alphas = self.alpha[Z]
        prior_logL = np.log(alphas)
        prior_logL = np.sum(prior_logL)

        logL = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                y_ij = self.A[i, j]
                p_ij = self.theta[Z[i], Z[j]]
                if y_ij == 1:
                    logL += np.log(p_ij)
                else:
                    logL += np.log(1 - p_ij)

        L = - (prior_logL + logL)
        return L

    def MAP_Z(self):
        return np.argmax(self.gamma, axis=1)

    def ICL(self):
        # maximize for model selection criterion
        ICL_a = -self.NLL()
        ICL_b = self.k*(self.k +1)  / 2
        ICL_b *= np.log(self.n * (self.n-1))
        ICL_c = (self.k - 1) * np.log(self.n)
        return ICL_a - 0.5 * (ICL_b - ICL_c)
        
