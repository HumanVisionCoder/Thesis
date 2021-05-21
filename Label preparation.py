# import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Import the excel file 
df = pd.read_csv("Annotated data SuJu11jun.csv")

#signer IDs
signers = df.iloc[:, 1]
signer_id = []

for ID in signers:
    if ID not in signer_id:
        signer_id.append(ID)
        

#make new df with signer,begin time, end time
newdf = df.drop(df.columns[[0,2,4,6,7,8]], axis = 1, inplace = False)

#here we specify the ammount of frames to be extracted from the ELAN files
ammount_of_frames_to_be_extracted = 900
#make rows for new dataframe
rows = []
for i in range(1,ammount_of_frames_to_be_extracted+1):
    rows.append(i)

#make a list of tuples with the signer,begin time, end time
timelist = []
newtimes = []
for index, row in newdf.iterrows():
    timelist.append((row[0],row[1],row[2]))
    
#We only want a certain ammoun frames and thus also a certain ammount of seconds
for times in timelist:
    if times[1]<= ammount_of_frames_to_be_extracted:
        newtimes.append(times)
        
#creates a list per signer containing frames they signed in  
def time_of_signing(signer):
    signer_times = []
    for times in newtimes:
        if times[0]==signer_id[signer]:
            signer_times.append((times[0],int(times[1]),int(times[2])))
    return signer_times

signer1_times = time_of_signing(0)
signer2_times = time_of_signing(1)

#get all the frames in which they are signing
def frames(signingtime):
    framessigner = []
    for times in signingtime:
        for i in range(times[1],times[2]+1):
            framessigner.append(i)
    return framessigner

framessigner1 = frames(signer1_times)
framessigner2 = frames(signer2_times)

#creates a list per signer giving as output a list of 0's and 1's 1 = signing 0 = not signing
def binary(frames):
    diarization = []
    for i in range(1,ammount_of_frames_to_be_extracted+1):
        if i in frames:
            diarization.append(1)
        else:
            diarization.append(0)
    return diarization

diarization1 = binary(framessigner1)
diarization2 = binary(framessigner2)


#function to create another list with 4 classes described in the function
def class_maker(a,b):
    class_column = []
    for i in range(0,len(a)):
        #both signers not signing
        if a[i] == 0 and b[i]== 0:
            class_column.append(0)
        #both signers siging
        if a[i] == 1 and b[i]== 1:
            class_column.append(3)
        #signer 1 signing 
        if a[i] == 1 and b[i] == 0:
            class_column.append(1)
        #signer 2 signing 
        if a[i] == 0 and b[i] == 1:
            class_column.append(2)
    return class_column

classes = class_maker(diarization1,diarization2)

#function to merge 3 lists
def merge_lists(a,b,c,d):
    mergeds = []
    for i in range(0, len(a)):
        merged = []
        merged.append(a[i])
        merged.append(b[i])
        merged.append(c[i])
        merged.append(d[i])
        mergeds.append(tuple(merged))
    return mergeds

signdiarization = merge_lists(rows,diarization1,diarization2,classes)
print(signdiarization)


#create columns for the new df
column = ["frames"]
column.extend(signer_id)
column.append("class")

#create new df
df = pd.DataFrame(signdiarization,
           columns=(column))

print(df)
df.to_csv(r'C:\Users\tobia\OneDrive\Documenten\thesis\SUJU11JULdataset.csv', index = False)


#%% visualisation %%#
#check how often signers sign at the same time 
same_time = df[(df.iloc[:,1] == 1) & (df.iloc[:,2] == 1)].index.tolist()
#both signers not signing 
not_signing = df[(df.iloc[:,1] == 0) & (df.iloc[:,2] == 0)].index.tolist()
#signer 1 signing but not signer 2
not_2 = df[(df.iloc[:,1] == 1) & (df.iloc[:,2] == 0)].index.tolist()
#signer 2 signing but not signer 1 
not_1 = df[(df.iloc[:,1] == 0) & (df.iloc[:,2] == 1)].index.tolist()


objects = ("Signer 1", "Signer 2","Both signing", "No siging")
y_pos = np.arange(len(objects))
performance = [len(not_2),len(not_1),len(same_time),len(not_signing)]
colours = ["g", "b", "c","r"]

plt.bar(y_pos, performance, align='center', alpha=0.5, color = colours)
plt.xticks(y_pos, objects)
plt.ylabel('Ammount of frames (1 fps)')
plt.title("Sign time distribution over "+str(ammount_of_frames_to_be_extracted) + " frames (1 fps)")

plt.show()