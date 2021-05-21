from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#we already know the shape of the data which is (576,576,3)
def grayscale(folder_with_images):
    filepath = glob(folder_with_images+"/*.jpg")
    arrays = []
    for img in filepath:
        #resize for less features
        img = Image.open(img).resize((50,50))
        img_gray =img.convert('L')
        img_array = np.array(img_gray)
        arrays.append(img_array)
    return arrays

#grayscaled images of signer 1 (left) and 2 (right) KN5JAN
KN5JAN_signer_1 = grayscale("KN5JAN signer 1")
KN5JAN_signer_2 = grayscale("KN5JAN signer 2" ) 
#grayscaled images of signer 1 (left) and 2 (right) REKE10JAN
REKE10JAN_signer_1 = grayscale("REKE10JAN signer 1" ) 
REKE10JAN_signer_2 = grayscale("REKE10JAN signer 2" ) 
#grayscaled images of signer 1 (left) and 2 (right) SUJU11JUN
SUJU11JUL_signer_1 = grayscale("SUJU11JUL signer 1" ) 
SUJU11JUL_signer_2 = grayscale("SUJU11JUL signer 2" ) 

def vectorize_img(arraylist):
        rescaled_imlist = []
        # Represent the image as a matrix of pixel weights, and flatten it
        for array in arraylist:   
            flattened_img = array.flatten()
            # Rescaling by dividing by the maximum possible value of a pixel
            #flattened_img = np.divide(flattened_img,255.0)
            rescaled_imlist.append(flattened_img)
        return rescaled_imlist

#vectors of images 
KN5JAN_signer_1 = vectorize_img(KN5JAN_signer_1 )
KN5JAN_signer_2 = vectorize_img(KN5JAN_signer_2)
#vectors of images
REKE10JAN_signer_1 = vectorize_img(REKE10JAN_signer_1)
REKE10JAN_signer_2 = vectorize_img(REKE10JAN_signer_2)
#vectors of images
SUJU11JUL_signer_1 = vectorize_img(SUJU11JUL_signer_1)
SUJU11JUL_signer_2 = vectorize_img(SUJU11JUL_signer_2)
#%%
"""performing PCA and appending PC's to the dataset """
def PCAs(vectorized_img, dataset):
    scaler = MinMaxScaler()
    #Use np.row_stack to transform a list of arrays to a pandas dataframe 
    data_rescaled = pd.DataFrame(np.row_stack(scaler.fit_transform(vectorized_img ))) #rescale data for PCA
    
    if str(vectorized_img).endswith("1"):
        signer = "S1 feature"
    else:
        signer = "S2 feature"
    x = data_rescaled.iloc[:, ].values
    pca = PCA(n_components= 100)
    principalComponents = pca.fit_transform(x)
    principal_Df = pd.DataFrame(data = principalComponents)
    columns = []
    for i in range(0,len(pca.explained_variance_ratio_)):
        columns.append(signer+" "+str(i))
    
    principal_Df.columns = columns
    
    #Append the columns of principal components to existing dataset with target variable
    principal_Df.to_csv(str(dataset)+'.csv', mode='a',header=columns)
    
    #Check the amount of explained variance per component
    principal_Df.tail()
    return 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)
    return len(pca.explained_variance_ratio_)

#PCAs(KN5JAN_signer_1,KN5JANdataset)
#PCAa(KN5JAN_signer_2,KN5JANdataset)

#PCAs(REKE10JAN_signer_1,REKE10JANdataset)
#PCAs(REKE10JAN_signer_2,REKE10JANdataset)

#PCAs(SUJU11JUL_signer_1,SUJU11JULdataset) 
#PCAs(SUJU11JUL_signer_2,SUJU11JULdataset) 

#%%PCA Plot of explained variance%##"
scaler = MinMaxScaler()
#Use np.row_stack to transform a list of arrays to a pandas datafra,e 
data_rescaled = pd.DataFrame(np.row_stack(scaler.fit_transform(KN5JAN_signer_2)))
x = data_rescaled.iloc[:,].values

pca_KN5JAN = PCA(n_components= 900)
principalComponents_KN5JAN = pca_KN5JAN.fit_transform(x)
principal_KN5JAN_Df = pd.DataFrame(data = principalComponents_KN5JAN)

plt.grid()
plt.plot(np.cumsum(pca_KN5JAN .explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('Scree plot.png')
