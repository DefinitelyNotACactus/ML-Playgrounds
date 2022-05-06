class SVM_Parameters:
  def __init__(self, svm_model, X, y):
    self.model = svm_model
    self.N = len(X)
    self.d = len(X[0])
    self.X = X
    self.y = y
    self.alpha = [0]*self.N
    self.somab_ = 0
    
    self.__separate_VS()
    self.compute_b()
    self.__comput_margin(X, y)

  def Kernel(self, xn, xm):
    xn = np.array(xn)
    xm = np.array(xm)
    if self.model.kernel == "rbf":
      vdiff = xn - xm
      return np.exp(-(1/self.model.n_features_in_) * np.sqrt(np.dot(vdiff, vdiff)) ** 2 )
    elif self.model.kernel == "poly":
      return ((1/self.model.n_features_in_) * np.dot(xn, xm) + self.model.coef0)**self.model.degree
    else:
      return np.dot(xn, xm)

  def compute_b(self):
    self.somab_ = 0
    for alpha, xm in zip(self.model.dual_coef_[0],self.model.support_vectors_):
      self.somab_ += alpha*self.Kernel(xm, self.xvs)
    
  def compute_left_hand(self, xn, yn):
    somaw_ = 0
    #somab_ = 0
    for alpha, xm in zip(self.model.dual_coef_[0],self.model.support_vectors_):
      somaw_ += alpha*self.Kernel(xm, xn)
      #somab_ += alpha*self.Kernel(xm, self.xvs)

    return yn*(somaw_ + 1/self.yvs - self.somab_)

  def print(self):
    for alpha, idvs in zip(self.model.dual_coef_[0], self.model.support_):
      print("Alpha: " + str(alpha) + " - ID: " + str(idvs))
  
  #Computa a menor distância média entre os VS das distintas classes
  def __comput_margin(self, X, y):
    soma_min_dist = 0
    for vs_0 in self.X_VS_0:
      min_dist = np.inf
      vs_0_array = np.array(vs_0)
      for vs_1 in self.X_VS_1:
        dist = np.linalg.norm(vs_0_array - np.array(vs_1))
        if(min_dist > dist):
          min_dist = dist
      soma_min_dist += min_dist

    self.margin = (soma_min_dist / len(self.X_VS_0))/2
    print("Margem: " + str(self.margin))

  def __separate_VS(self):
    self.xvs = None
    self.yvs = None
    self.X_VS_0 = []
    self.X_VS_1 = []
    vs_no_margin = []
    for c, x, id_vs in zip(self.model.dual_coef_[0], self.model.support_vectors_, self.model.support_):
      alpha_ = abs(c)
      self.alpha[id_vs] = alpha_
      if alpha_ > 0 and alpha_ < self.model.C: #Teste se é VS na margem
        #Pega um VS na margem qualquer
        if(self.yvs == None):
            self.xvs = self.X[id_vs]
            self.yvs = self.y[id_vs]
            return
        if c < 0:
          self.X_VS_0.append(x.tolist())
        else:
          self.X_VS_1.append(x.tolist())
      else:
        vs_no_margin.append(id_vs)
    
    #Remoção das amostras, classificadas como VS fora da margem, de X e y
    X = []
    y = []
    for id in range(len(self.X)):
      if id not in vs_no_margin:
        X.append(self.X[id])
        y.append(self.y[id])
    self.X = X
    self.y = y
    self.N = len(X)
