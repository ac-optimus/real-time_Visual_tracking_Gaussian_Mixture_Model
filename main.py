from gmm import GMM
PATH = "demo_vid.avi"
PATH_test = PATH
#hyper parameters
train_frame_cnt  = 50  #number of frames to train on
K = 4
alpha = 0.05
T_b = 0.90
threshVar=2.5
#define model
gmm = GMM(K, alpha, T_b, threshVar)
print ("initialized successfully")
print (gmm.weights)
print ("training start...")
#train
gmm.perform(PATH, train_frame_cnt, task = "train")
print ("done with training")
#test
gmm.perform(PATH_test, train_frame_cnt, task="test")
print ("testing done!")