import numpy as np
import cv2
from scipy.stats import multivariate_normal
import time
import sys

class Gaussian:
    """gaussian distribution for a given mean and standard deviation"""
    def __init__(self, mean,std):
        self.mean = [mean] if type(mean) != list else mean
        self.std = [std] if type(std) != list else std
    
    def sample(self):
        """"samples a point randomly from the gaussian"""
        std, mean = self.std, self.mean
        samp = np.random.Gaussian(mean, std)
        return samp

    def check(self, thresh, pnt):
        """checks if the current pixel belong to this gaussian 
        thresh --> threshold applied to classify if a point belongs to this gaussian
        pnt --> current pixel value/ vector
        """
        pnt = [pnt] if type(pnt) != list else pnt 
        std = self.std[0] #assumption of same varience in rbg
        check_flag = 0
        for pnt_i in pnt:
            if np.negative(std)*thresh <= pnt_i <= std*thresh:
                pass
            else:
                check_flag = 1
        if check_flag == 1:
            return False
        else:
            return True

    def prob(self, pnt):
        """returns the probability(pdf) for a pixel"""
        pnt = np.asarray([pnt]) if type(pnt)!=list else pnt
        dim = len(self.mean)
        mean = np.asarray(self.mean)
        std = self.std[0]
        cov = np.square(std)*np.eye(dim, dim)
        cov_inverse = (1/np.square(std))*np.eye(dim, dim)
        prob_x = (1/((2*np.pi)**(dim/2)* np.linalg.det(cov)))* \
                np.exp((-1/2)*(np.transpose(pnt-mean)*cov_inverse*(pnt-mean)))
        return prob_x[0]
        


class GMM:
    def __init__(self, K, alpha, T_b, threshVar=2.5):
        self.Distributions = []#list of all the gaussian in the gaussian mixture model
        self.weights = []
        self.K = K #number of clusters to use
        self.init_GMM()
        self.alpha = alpha #learning rate
        self.threshVar = threshVar #threshold to check the pixel belonging
        self.T_b = T_b #for background substraction
        self.B=  None #top B gaussians representing background
    def init_GMM(self):
        """
        initialize the GMM model
        """
        for model_k in range(self.K):
            #see the hardcoded values
            new_model = Gaussian(0, 1)#mean ==0, std==1
            self.Distributions.append(new_model)
        
        self.weights = [0.7, 0.11, 0.1, 0.09]

    def perform(self, path, train_frame_count, task ="train"):
        """
            combined method for train and test
            task --> train/ test
            path --> path to the video file
        """
        cap = cv2.VideoCapture(path)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        # Read until video is completed
        process_frameCnt, count = int(train_frame_count), 0
        while(cap.isOpened()) and count !=process_frameCnt:
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_size = frame.shape
                gray = cv2.resize(frame,(128, 72))  #resize the image
                #normalize the frame
                norm_image = cv2.normalize(gray, None, \
                    alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
                
                if task == "train":
                    self.add_frame(norm_image)
                    print ("frame count-->",count)
                    count += 1

                else:
                    self.get_background() #get the top B gaussians for background
                    frame = self.rmvBckgrd(norm_image)
                    cv2.imshow("trained on %s frames"%train_frame_count, frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

    def add_frame(self, curFrame):
        """add current frame to the existing GMM"""
        for row_i in range(curFrame.shape[0]):
            for column_j in range(curFrame.shape[1]):
                #each pixel value 
                pixel_i = curFrame[row_i][column_j].tolist()\
                     if type(curFrame[row_i][column_j]) != list\
                          else [curFrame[row_i][column_j]]

                check_flag = 0 #weather pixel matches any of the gaussian
                index_lst = []#index of all the matched gaussian
                for model_k in self.Distributions:
                    if model_k.check(self.threshVar, pixel_i):
                        #update this model
                        check_flag = 1
                        index_model_k = self.Distributions.index(model_k)
                        index_lst.append(index_model_k)
                        self.updt_model(model_k, pixel_i)
                if check_flag == 0:
                    #add new gaussian, replace the least weight one
                    self.rplc_gaus(pixel_i)
                self.updt_wgt(index_lst)
    
    def updt_model(self, model_k, pixel):
        """update mean and std of the model_k"""
        pixel = np.asarray([pixel]) if type(pixel)!=list \
                            else np.asarray(pixel)
        #second learning rate
        lr_row = self.alpha*model_k.prob(pixel.tolist())
       
        mu_prev = np.asarray(model_k.mean)#mean must be a list(vector)
        std_prev = np.asarray(model_k.std)
        #new mean and std
        mu = (1-lr_row)*mu_prev + lr_row*pixel
        std_sq = (1-lr_row)*std_prev**2 +\
             lr_row*(np.linalg.norm(pixel - mu)**2)
        #update the parameters
        model_k.std = np.sqrt(std_sq)
        model_k.mean = mu
        return 0


    def updt_wgt(self, index_lst):
        """update the model weight
           index_lst --> list of index of matched gaussians
        """
        wgt_prev = np.asarray(self.weights)
        M_k =  np.zeros(wgt_prev.shape)
        M_k[index_lst] = 1 #if model matches
        wgt_new = (1-self.alpha)*wgt_prev + self.alpha*M_k
        #normalize the weight
        wgt_new = wgt_new/wgt_new.sum()
        #update weight 
        self.weights = wgt_new.tolist()
        return 0 

    def rplc_gaus(self, pixel_i):
        """replace the smallest model by new model
           pixel_i --> 3d for rgb and 1d for gray scale
        """
        pixel_i = [pixel_i] if type(pixel_i) != list else pixel_i
        sorted_wgt = sorted(self.weights)
        index_least = self.weights.index(sorted_wgt[0])
        mu, std = pixel_i, [1]*len(pixel_i)#very high varience
        new_model = Gaussian(mu, std)#new gaussian
        #update the weights and replace new model
        self.weights[index_least] = 0.09#low weight
        self.Distributions[index_least] = new_model
        return 0

    def get_background(self):
        """get the gaussians that represents the background"""
        w_by_std = [self.weights[i]/self.Distributions[i].std[0]\
                        for i in range(len(self.Distributions))]
        sorted_w_by_std = sorted(w_by_std)[::-1]
        sum_ = 0
        index = []
        for i in range(len(w_by_std)):
            if sorted_w_by_std[i] > self.T_b:
                pass
            elif sum_ < self.T_b:
                index.append(i)
                sum_+=sorted_w_by_std[i]
        B = np.asarray(self.Distributions)[index]
        self.B = B.tolist()
        #background model ready till here
        return 0
    def rmvBckgrd(self, curFrame):
        """assign pixel as foreground(white) or backgroung(black)"""
        B = self.B
        for row_i in range(curFrame.shape[0]):
            for column_j in range(curFrame.shape[1]):
                pixel_i = curFrame[row_i][column_j].tolist() if type(curFrame[row_i][column_j]) != list else [curFrame[row_i][column_j]]
                check_flag = 0
                for model_k in B:
                    if model_k.check(self.threshVar, pixel_i):
                        #classify as background --> black
                        curFrame[row_i][column_j] = 0 
                        check_flag=1
                if check_flag == 0:
                    #classify as forground --> white
                    curFrame[row_i][column_j] = 255
        return curFrame