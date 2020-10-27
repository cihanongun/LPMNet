import numpy as np
import json

url = "https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip"
folder_path = "data/shapenetcore_partanno_segmentation_benchmark_v0/"

# Random downsampling or zero-padding to set all models to the same size
def get_models(file_names, cat_number, set_size = 2048):
    all_models = []

    for m in file_names:
        
        model_file = []

        with open(folder_path + cat_number + "/points/" + m + ".pts") as f:
            i=0
            for line in f: 
                model_file.append([float(x) for x in line.split()]) 
                i = i+1

        PCarray = np.array(model_file)
        label_file = []

        with open(folder_path + cat_number + "/points_label/" + m + ".seg") as f:
            for line in f:
                label_file.append([float(x) for x in line.split()])
                
        PClabel = np.array(label_file)#[random_points]
        last_model = np.concatenate((PCarray,PClabel),axis=1)
        
        
        if(last_model.shape[0] > set_size):
            point_size = len(model_file)
            random_points = np.random.choice(point_size, set_size, replace=False)
            last_model = last_model[random_points]
            
        elif (last_model.shape[0] < set_size):
            padded_model = np.zeros([set_size,4])
            for i in range(len(last_model)):
                padded_model[i] = last_model[i]
            last_model = padded_model
        
        all_models.append(last_model)
    
    model_array = np.array(all_models)
    return model_array

def get_dataset(category ="Chair", split = "train", point_size = 2048):
    
    category_list = []
    cat_number = 0
    with open(folder_path + "synsetoffset2category.txt", 'r') as f:
        for line in f:
            if category in line:  
                cat_number = line.replace(category+"\t" , "")[:-1]
    
    if(cat_number == 0) :
        raise ValueError("Wrong category")
        return
    
    if(split == "train"):
        model_names = json.load(open(folder_path + "train_test_split/shuffled_train_file_list.json", 'r'))
    elif(split == "test"):
         model_names = json.load(open(folder_path + "train_test_split/shuffled_test_file_list.json", 'r'))
    elif(split == "validation"):
         model_names = json.load(open(folder_path + "train_test_split/shuffled_val_file_list.json", 'r'))
    else :
        raise ValueError("Wrong set type!")
        return
    
    cat_names = []
    for m in model_names:
        if cat_number in m:
            cat_names.append(m.replace("shape_data/" + cat_number + "/" , ""))
            
    return get_models(cat_names, cat_number, point_size)