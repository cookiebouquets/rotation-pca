from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class dorkPCA(BaseEstimator, TransformerMixin):
    def __init__(self, 
                    D, d,
                    eta=0.1, mu_learning_factor=0.02, Gamma_learning_factor=0.05, evals_learning_factor=0.005,
                    rotation=True, rotation_factor=0.002,
                    Psitype=0, beta=0.1, ksi=0.1,
                    kernel_function=None, U=None, rbf_gamma=1.0, poly=2):
        
        # Initialize hyperparameters here

        # shape of the data to work with
        self.D = D # dimension of each data vector x
        self.M = U.shape[0] if U is not None else D # dimension of each kernel datum h
        self.d = d # dimension of subspace to estimate

        # learning rates
        self.eta = eta # overall learning rate
        self.mu_learning_factor = mu_learning_factor # relative learning rate for mu
        self.mu_learning_rate = self.eta*self.mu_learning_factor # overall learning rate for mu
        self.Gamma_learning_factor = Gamma_learning_factor # relative learning rate for Gamma
        self.Gamma_learning_rate = self.eta*self.Gamma_learning_factor # overall learning rate for Gamma
        self.evals_learning_factor = evals_learning_factor # relative learning rate for evals
        self.evals_learning_rate = self.eta*self.evals_learning_factor # overall learning rate for evals
        self.rotation_factor = rotation_factor # relative learning rate for rotation

        # params related to robust loss function Psi
        self.Psitype = Psitype # the type of loss function to use (0,1,2, ...)
        self.beta = beta # tuning parameter, used for types 1 and 2
        self.ksi = ksi # tuning parameter, used for type 2

        self.rotation = rotation # whether to rotate the subspace
        
        # params related to kernelization
        self.kernel_function = kernel_function # None if no kernel, otherwise string
        self.U = U # list of basis vectors for kernel calculation (M, D)
        self.rbf_gamma = rbf_gamma # parameter for RBF kernel function
        self.poly = poly # degree of polynomial kernel function
        
        
        # Initialize internal variables here

        # means, Gamma, evals
        self.mu_ = np.zeros(self.M) # only gets initialized the first time the model is called
        self.Gamma_ = np.eye(self.M, d) # only gets initialized the first time the model is called
        self.evals_ = np.zeros(d) # only gets initialized the first time the model is called

    # internal helper functions (start with underscore by convention)
    # function to calculate coefficients
    def _loadings(self, h_x):
        """
        Compute the coefficients of a vector h_x - mu along the subspace spanned by the columns of Gamma.

        Parameters:
        - h_x: kernel datum vector (dimension M).
        - mu: mean vector (dimension M).
        - Gamma: matrix of basis vectors (dimension (M , d)).

        Returns:
        - loadings: vector with coefficients, one for each of the basis vectors in Gamma (dimension d).
        """
        loading = (h_x - self.mu_) @ self.Gamma_

        return loading

    # function to calculate the z-distance
    def _z_distance(self, h_x):
        """
        Compute the z-distance of a vector h_x - mu to the subspace spanned by the columns of Gamma.

        Parameters:
        - h_x: kernel datum vector (dimension M).
        - mu: mean vector (dimension M).
        - Gamma: matrix of basis vectors (dimension (M , d)).

        Returns:
        - z-distance (scalar).
        """
        # project h_x - mu onto the columns of Gamma (in steps, to avoid huge matrices)
        v = self.Gamma_.transpose().dot(h_x - self.mu_)
        v = self.Gamma_.dot(v)
        res_vector = h_x - self.mu_ - v
        # calculate half the norm-squared of the res_vector
        z = (res_vector.transpose() @ res_vector) / 2
        return z

    # function to calculate matrix of Lagrange multipliers
    def _Lagrange_matrix(self, h_x):
        """
        Compute the Lagrange multipliers.
        Note: This does not yet include the effect of the weights dPsi.

        Parameters:
        - h_x: kernel datum vector (dimension M).
        - mu: mean vector (dimension M).
        - Gamma: matrix of basis vectors (dimension (M , d)).

        Returns:
        - Lagrange multipliers (d x d matrix). The lower triangular has already been set to zero.

        """
        return np.triu(np.outer(self.Gamma_.transpose()@(h_x-self.mu_),((h_x-self.mu_)@self.Gamma_)))

    def _grad_mu(self, h_x, w):
      return -w*(h_x-self.mu_)

    def _grad_Gamma(self, h_x, Lagrange, w):
      return -w*(np.outer((h_x-self.mu_),(h_x-self.mu_)@self.Gamma_)-self.Gamma_@Lagrange)

    # function to calculate the weight based on robust loss function Psi
    def _dPsi(self, z):
        """
        dPsi is the derivative of the non-linear robust loss function Psi
        It assigns smaller weights to greater distances z

        Parameters:
        - z: should be the distance from vector to subspace (positive scalar)
        - Psitype: the type of loss function to use (0,1,2, ...)
        - beta: tuning parameter, used for types 1 and 2
        - ksi: tuning parameter, used for type 2

        Returns:
        - dPsi(z): weight calculated from distance (assigns smaller weight to greater distance) (positive scalar)
        """
        if self.Psitype == 0:   # this is the conventional PCA, no robustness
          dPsi = 1
        elif self.Psitype == 1:
          dPsi = np.exp(-self.beta * z)
        elif self.Psitype ==2:
          dPsi = 2 - 2/(1+np.minimum(1,np.exp(-self.beta*(z-self.ksi))))

        return dPsi


    # rotation matrix
    def make_rotation_matrix(self,v,w,alpha):
      """
      Make a rotation matrix that rotates in the plane between v and w
      by an angle that is a proportion alpha of the original angle.
      The direction of rotation will be from v to w

      Parameters
      ----------
      v, w : (numpy arrays) The two vectors that define the plane.
      alpha : (scalar) The proportion of the original angle that we want to rotate through.

      Returns
      -------
        The rotation matrix.
      """

      D = len(v)

      # make a matrix that has v, w as first two columns
      # and random columns to make it into a square matrix
      S = np.vstack([v,w, np.random.rand(D-2,D)]).T

      # orthonormalize:
      Q, _ = np.linalg.qr(S)
      # this normalizes the first column
      # first and second column still span the original plane of (v,w)
      # the remaining columns are an orthonormal basis for the orthogonal complement of (v,w)

      # check the residue of w after projecting onto v
      wp = np.dot(v,w)/np.dot(v,v)*v
      wq = w-wp

      # check if the signs are ok
      if np.sign(wp @ Q[:,0]) != np.sign(wq @ Q[:,1]):
        Q[:,1] = -Q[:,1] # flip the sign of the second column of Q

      # find the original angle theta between (v,w)
      ratio = np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w))
      if np.abs(ratio) >= 1:
        theta = 0 # if there is something wrong with the ratio, just don't rotate
      else:
        theta = np.arccos(ratio)

      # calculate the final rotation angle phi that we want to rotate by
      phi = alpha*theta

      # 2x2 rotation matrix through an angle phi
      R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

      # make this into a DxD matrix, filling in with identity matrix
      B = np.eye(D)
      B[:2,:2] = R

      # return the final rotation matrix
      return Q @ B @ Q.T

    # process one sample of data
    def _fit_one(self, x):
        # if kernel function is called, first change x into h using compute_kernel_datum
        if self.kernel_function is None:
            h = x
        else:
            h = self.compute_kernel_datum(x, self.U, self.kernel_function, self.rbf_gamma, self.poly)

        # calculate distance and weight
        z = self._z_distance(h)
        weight = self._dPsi(z)

        # update mean estimate mu
        g_mu = self._grad_mu(h, weight)
        self.mu_ = self.mu_ - self.mu_learning_rate*g_mu

        # calculate eigenvalues
        self.evals_ = (1-self.evals_learning_rate*weight)*self.evals_ + self.evals_learning_rate*weight*(self._loadings(h))**2

        # sort the Gamma vectors
        self.Gamma_ = self.Gamma_[:, np.argsort(self.evals_)[::-1]] # sort the columns of Gamma based on evals
        self.evals_ = np.sort(self.evals_)[::-1] # also sort the evals themselves

        if(self.rotation):
            # prepare for rotation from the closest vector of Gamma towards the data sample
            # find the vector closest to the new data vector
            v = self.Gamma_[:,np.argmax(np.abs((h-self.mu_)@self.Gamma_))] # pick the vector from Gamma closest to h-mu
            w = h-self.mu_

            beta = self.rotation_factor*weight # the proportion that we want to rotate by
            A = self.make_rotation_matrix(v,w,beta) # rotate from v to h-mu by a portion beta
            self.Gamma_ = A @ self.Gamma_ # apply the rotation matrix to Gamma

            # another rotation, this time from the projection onto Gamma towards the data sample
            v = self.Gamma_ @ self.Gamma_.T @ (h-self.mu_) # the projection of h-mu onto Gamma
            beta = self.rotation_factor*weight # the proportion that we want to rotate by
            A = self.make_rotation_matrix(v,w,beta) # rotate from v to h-mu by a portion beta
            self.Gamma_ = A @ self.Gamma_ # apply the rotation matrix to Gamma


        # calculate Lagrange multipliers and gradient of loss function
        Lagrange = self._Lagrange_matrix(h)
        g_Gamma = self._grad_Gamma(h, Lagrange, weight)

        # update Gamma based on gradient
        self.new_Gamma_ = (1-self.Gamma_learning_rate)*self.Gamma_ - self.Gamma_learning_rate*g_Gamma/((np.linalg.norm(h-self.mu_))**2) # normalize here, helps with convergence
        # enforce that Gamma remains orthonormal
        self.new_Gamma_, _ = np.linalg.qr(self.new_Gamma_)
        # check if no columns of Gamma have abruptly flipped dircetion
        for i in range(self.d):
          if self.Psitype != 0 and self.new_Gamma_[:,i] @ self.Gamma_[:,i] < 0:
            self.Gamma_[:,i] = -self.new_Gamma_[:,i]
          else:
            self.Gamma_[:,i] = self.new_Gamma_[:,i]


        return self

    def fit(self, X, y=None):
        # if X is one sample, process it
        # if X is a 2D array, process each sample
        if X.ndim != 2:
          self._fit_one(X)
        else:
          for x in X:
            self._fit_one(x)
        return self

    # transform one sample of data
    def _transform_one(self, x):
        # if kernel function is called, first change x into h using compute_kernel_datum
        if self.kernel_function is None:
            h = x
        else:
            h = self.compute_kernel_datum(x, self.U, self.kernel_function, self.rbf_gamma, self.poly)
      
        # Apply the transformation logic
        return (h-self.mu_)@self.Gamma_

    def transform(self, X):
        # if X is one sample, process it
        # if X is a 2D array, process each sample
        if X.ndim != 2:
          return self._transform_one(X)
        else:
          coefs = np.zeros_like(len(X), self.d)
          for i, x in enumerate(X):
            coefs[i] = self._transform_one(x)
        return coefs
    
    # implement poly kernel function:
    def poly_kernel(self, x1, x2, poly=2):
        """
        Compute the polynomial kernel between two vectors x1 and x2.
    
        Parameters:
        - x1, x2: Input vectors.
        - poly: Degree of the polynomial.
    
        Returns:
        - dot product raised to power of poly (scalar).
        """
        return (np.dot(x1, x2))**poly # +1 or +c causes it to include lower degrees as well
    
    # implement rbf kernel function:
    def rbf_kernel(self, x1, x2, rbf_gamma=1.0):
        """
        Compute the RBF kernel between two vectors x1 and x2.
    
        Parameters:
        - x1, x2: Input vectors.
        - rbf_gamma: Kernel parameter.
    
        Returns:
        - Kernel value (scalar).
        """
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.exp(-rbf_gamma * squared_distance)
    
    # implement dot kernel function:
    def dot_kernel(self, x1, x2):
        """
        Compute the dot product as kernel between two vectors x1 and x2.
        This is equivalent to no kernel function.
        Parameters:
        - x1, x2: Input vectors.
    
        Returns:
        - dot product (scalar).
        """
        return np.dot(x1, x2)
    
    # calculate kernel datum relative to some basis U
    def compute_kernel_datum(self, x, U, kernel_function, rbf_gamma=1.0, poly=2):
        """
        Compute the kernel matrix datum for a vector x relative to a basis U using a given kernel function.
    
        Parameters:
        - x: input vector (dimension D).
        - U: list of basis vectors (dimension (M, D)).
        - kernel_function: Kernel function to use.
        - rbf_gamma: Kernel parameter used for the RBF kernel.
        - poly: Degree of the polynomial kernel.
    
        Returns:
        - kernel datum (dimension M).
        """
        M = U.shape[0]
        kernel_datum = np.zeros(M)
    
        for i in range(M):
            if kernel_function == 'rbf_kernel':
              kernel_datum[i] = self.rbf_kernel(x, U[i], rbf_gamma)
            elif kernel_function == 'dot_kernel':
              kernel_datum[i] = self.dot_kernel(x, U[i])
            elif kernel_function == 'poly_kernel':
              kernel_datum[i] = self.poly_kernel(x, U[i], poly)
            else:
              raise ValueError("Invalid kernel function")
    
        return kernel_datum
        
# function to calculate relative error between subspaces spanned by two bases
def grassmann_distance(X0, X1):
  """
  Calculate Grassmann Distance between subspaces spanned by two bases.

  Inputs:
  - X0: np.array of shape (D, d) whose columns are the basis vectors for the first subspace
  - X1: np.array of shape (D, d) whose columns are the basis vectors for the second subspace

  Note:
  - D: dimension of the overall data
  - d: dimension of the subspace

  Output:
  - relative error between the two subspaces, calculated as the RMS of the principal angles between the two bases

  """
  # calculate X0^T X1
  R = np.dot(X0.T.conj(), X1)
  # calculate SVD to get singular values
  _, S, _ = np.linalg.svd(R, full_matrices=True)
  S[np.where(S > 1)] = 1.0
  # get angles
  theta = np.arccos(S)
  # square add the angles
  distance = np.sqrt(np.mean(np.sin(theta) ** 2))

  return distance

# compare similarity of two vectors
def cosine_similarity(v, w):
  return np.abs(np.real(np.dot(v, w)) / (np.linalg.norm(v) * np.linalg.norm(w)))

# function to calculate relative error between two subspaces, weighted for eigenvalues
def var_grassmann_distance(X0, X1, evals):
  """
  Calculate Grassmann Distance between subspaces spanned by two bases, weighted for eigenvalues

  Inputs:
  - X0: np.array of shape (D, d) whose columns are the basis vectors for the first subspace
  - X1: np.array of shape (D, d) whose columns are the basis vectors for the second subspace
  - evals: np.array of shape (d) of the eigenvalues of the data

  Note:
  - D: dimension of the overall data
  - d: dimension of the subspace

  Output:
  - relative error between the two subspaces, calculated as the RMS of the principal angles between the two bases, weighted for eigenvalues

  """
  # calculate X0^T X1
  R = np.dot(X0.T, X1)
  # calculate SVD to get singular values
  _, S, _ = np.linalg.svd(R, full_matrices=True)
  S[np.where(S > 1)] = 1.0
  # get angles
  theta = np.arccos(S)
  # square add the angles

  distance = np.sqrt(np.sum((np.abs(evals)*np.sin(theta))**2)/np.sum(np.abs(evals)**2))

  return distance