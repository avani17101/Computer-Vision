import numpy as np
import maxflow
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def display_img_arr(img_arr, r, c, dim,titles_arr):
    fl = 0
    fig = plt.figure(figsize = dim)
    for i in range(r):
        for j in range(c):
            if len(img_arr) == fl:
                break
            ax1 = fig.add_subplot(r, c, fl + 1)
            ax1.set_title(titles_arr[fl], fontsize = 20)
            ax1.imshow(img_arr[fl], cmap = 'gray')
            fl = fl + 1
    plt.show()
    
class GrabCut:
    def __init__(self, img, gamma, k, connected, max_iter, bbox, mask, iters):
        h = img.shape[1]
        w = img.shape[0]

        self.k = k
        self.gamma = gamma
        self.img = img.astype(np.float64)
       
        up = ((self.img[1:, :] - self.img[:-1, :])**2).sum()# Up-difference expectation
        upright = ((self.img[1:, :-1] - self.img[:-1, 1:])**2).sum() # Up-Right difference  expectation
        left = ((self.img[:, 1:] - self.img[:, :-1])**2).sum() # Left-difference expectation
        upleft = ((self.img[1:, 1:] - self.img[:-1, :-1])**2).sum() # Up-Left difference expectation
        
        
        g = maxflow.GraphFloat()
        self.nodeids = g.add_grid_nodes(self.img.shape[:2])
        
        tot_pixels = 4*w*h - 3*h - 3*w + 2
        inv_beta = up + upright + left + upleft
        inv_beta = 2*inv_beta/tot_pixels
        self.beta = 1/inv_beta  # according to formula given in paper

        for x in range(h):
            for y in range(w):
                source_node = self.nodeids[y, x]
                if y-1 >= 0: # if top neighbor exists
                    dest_node = self.nodeids[y-1, x]
                    temp = np.sum((self.img[y, x] - self.img[y-1, x])**2)
                    wt = np.exp(-self.beta * temp)
                    
                    n_link = self.gamma/1 * wt
                    source_node = self.nodeids[y, x]
                    g.add_edge(source_node, dest_node , n_link, n_link)

                if x-1 >= 0: # if left neighbor exists
                    dest_node = self.nodeids[y, x-1]
                    temp = np.sum((self.img[y, x] - self.img[y, x-1])**2)
                    wt = np.exp(-self.beta * temp)
                    n_link = self.gamma/1 * wt
                    source_node = self.nodeids[y, x]
                    g.add_edge(source_node, dest_node, n_link, n_link)
                
                if connected == 8:
                    if x-1 >= 0 and y-1 >= 0: # if top left neighbor exists
                        dest_node  = self.nodeids[y-1, x-1]
                        temp = np.sum((self.img[y, x] - self.img[y-1, x-1])**2)
                        wt = np.exp(-self.beta * temp )
                        source_node = self.nodeids[y, x]
                        n_link = self.gamma/np.sqrt(2) * wt
                        g.add_edge(source_node, dest_node, n_link, n_link)
                    
                    if x+1 < h and y-1 >= 0: # if top right neighbor exists
                        source_node = self.nodeids[y, x]
                        temp = np.sum((self.img[y, x] - self.img[y-1, x+1])**2)
                        wt = np.exp(-self.beta * temp)
                        n_link = self.gamma/np.sqrt(2) * wt
                        dest_node = self.nodeids[y-1, x+1]
                        g.add_edge(source_node, dest_node, n_link, n_link)
        
        self.graph = g
        self.GC_BGD = 0 # Sure Background
        self.GC_FGD = 1 # Sure Foreground
        self.GC_PR_BGD = 2  #Probable background
        self.GC_PR_FGD = 3 #Probable foreground

        self.bbox = bbox
        x, y, w, h = bbox
        temp  = np.ones(shape = self.img.shape[:2])

       
        self.mask = self.GC_BGD * temp # Initially all trimap background
        self.mask[np.where(mask == 0)] = self.GC_BGD # Sure background
        left_start = y
        left_end = y+h+1
        right_start = x
        right_end = x+w+1

        self.mask[left_start:left_end, right_start:right_end] = self.GC_PR_FGD # trimap unknown set
        self.mask[np.where(mask == 1)] = self.GC_FGD # Sure foreground
        
        
        r_ind, c_ind = np.where(self.mask == self.GC_FGD)

        for i in range(len(r_ind)):
            r = r_ind[i]
            c = c_ind[i]
            edge = self.nodeids[r, c]
            self.graph.add_tedge(edge, np.inf, 0)
        
        r_ind, c_ind = np.where(self.mask == self.GC_BGD)
        
        for i in range(len(r_ind)):
            r = r_ind[i]
            c = c_ind[i]
            edge = self.nodeids[r, c]
            self.graph.add_tedge(edge, 0, np.inf)
        

        for i in range(iters):
            print("iteration "+str(i)+'\n')
            bg_condition = np.logical_or(self.mask == self.GC_BGD, self.mask == self.GC_PR_BGD)
            bg_pix = self.img[np.where(bg_condition)]
            bg_ci = np.empty(shape = len(bg_pix), dtype = int)
            KMB = KMeans(n_clusters=self.k, max_iter = max_iter).fit(bg_pix) # K Means for background pixels
        
            self.GMM_bg = GaussianMixture(n_components = self.k)
            self.GMM_bg.fit(bg_pix, KMB.labels_)

            fg_condition = np.logical_or(self.mask == self.GC_FGD, self.mask == self.GC_PR_FGD)
            fg_pix = self.img[np.where(fg_condition)]
            fg_ci = np.empty(shape = len(fg_pix), dtype = int)
            KMF = KMeans(n_clusters=self.k, max_iter = max_iter).fit(fg_pix) # K Means for foreground pixels
            self.GMM_fg = GaussianMixture(n_components = self.k)
            self.GMM_fg.fit(fg_pix, KMF.labels_)
           
            bg_ci = self.GMM_bg.predict(bg_pix)
            fg_ci = self.GMM_fg.predict(fg_pix)

            self.GMM_bg.fit(bg_pix, bg_ci)
            self.GMM_fg.fit(fg_pix, fg_ci)

            D_initial_bg = self.GMM_bg.weights_ / np.sqrt(np.linalg.det(self.GMM_bg.covariances_))
            D_initial_fg = self.GMM_fg.weights_ / np.sqrt(np.linalg.det(self.GMM_fg.covariances_))
            cov_inv_fg =  np.linalg.inv(self.GMM_fg.covariances_)
            cov_inv_bg =  np.linalg.inv(self.GMM_bg.covariances_)
            t_links_bg = np.empty(shape = (self.img.shape[0],self.img.shape[1]),dtype = np.float64)
            t_links_fg = np.empty(shape = (self.img.shape[0],self.img.shape[1]), dtype = np.float64)
            
            unkonwn_ind = np.logical_or(self.mask == self.GC_PR_BGD, self.mask == self.GC_PR_FGD)
            r_ind, c_ind = np.where(unkonwn_ind)
            
            for k in range(len(r_ind)):
                node = self.img[r_ind[i], c_ind[i]]
                summ = 0
                summ2 = 0
                for i in range(self.k):
                    summ += D_initial_bg[i] * np.exp(-0.5 * (self.img[r_ind[k], c_ind[k]] - self.GMM_bg.means_[i]).reshape(1, 3) @ cov_inv_bg[i] @ (self.img[r_ind[k], c_ind[k]] - self.GMM_bg.means_[i]).reshape(3, 1))[0][0] 
                    summ2 += D_initial_fg[i] * np.exp(-0.5 * (self.img[r_ind[k], c_ind[k]] - self.GMM_fg.means_[i]).reshape(1, 3) @ cov_inv_fg[i] @ (self.img[r_ind[k], c_ind[k]] - self.GMM_fg.means_[i]).reshape(3, 1))[0][0]

                t_links_fg[r_ind[k], c_ind[k]] = -np.log(summ)
                t_links_bg[r_ind[k], c_ind[k]] = -np.log(summ2)

                self.graph.add_tedge(self.nodeids[r_ind[k], c_ind[k]], t_links_fg[r_ind[k], c_ind[k]], t_links_bg[r_ind[k], c_ind[k]])
            
           
            self.graph.maxflow()
          
            for i in range(len(r_ind)):
                edge = self.nodeids[r_ind[i], c_ind[i]]
                self.graph.add_tedge(edge, -t_links_fg[r_ind[i], c_ind[i]], -t_links_bg[r_ind[i], c_ind[i]])
                
                if self.graph.get_segment(edge) == 0:
                    self.mask[r_ind[i], c_ind[i]] = self.GC_PR_FGD
                else:
                    self.mask[r_ind[i], c_ind[i]] = self.GC_PR_BGD
