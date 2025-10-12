import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import pandas as pd
from load_datadir_re  import load_datadir_re
from myPMS import myPMS
from myPCA import myPCA
from myRobustPMS import myRobustPMS
from mynormal2depth import mynormal2depth, surface2mesh


def save_results(Normal, object_name_list:list, Id:int, method:str, size):
    """ Save Normal vector graph and the mat file
    ## Parameters:
    -   Normal:
            normal map of object
    -   object_name_list:
            the list of objects' names
    -   Id:
            current object's ID
    -   method: 
            methods and hyper-parameters used
    -   size:
            image size
    """

    plt.figure(figsize=(6, 6*size[1]/size[0]))
    # Because the range of each element in the unit vector is [-1,1], 
    # we scale it to the RGB's elements range [0,1]
    plt.imshow((Normal+1.0)/2)
    plt.xlabel("pixel x")
    plt.ylabel("pixel y")
    plt.title(f"Normal Vectors of {object_name_list[Id]} Surface\n using {method}")
    plt.savefig(f"../results/{method}_{Id+1}.png",
                dpi = 400)
    
    # Saving results to .mat form
    # normal_vectors = {"normal vectors":Normal}
    # savemat(f"../results/{method}_normal_vectors_{Id+1}.mat", normal_vectors)

    # Saving results to Excel form
    excel_writer = pd.ExcelWriter(f'../results/{method}_normal_vectors_{Id+1}.xlsx',
                                   engine='openpyxl')
    channels = ['X', 'Y', 'Z']
    for i, channel_name in enumerate(channels):
        channel_df = pd.DataFrame(Normal[:, :, i])
        channel_df.to_excel(excel_writer, sheet_name=channel_name, index=False, header=False)

    excel_writer.close()







def sphere_normals(image_size, radius, center=None):
    """
    Function to get reference sphere normals

    ## Parameters:
    -   image_size:
            size of the original image
    -   radius:
            radius of sphere in pixel
    -   center: 
            center of sphere in pixel
    
    ## Returns
    -   Normal map of sphere 
    """

    height, width = image_size
    if center is None:
        center = (height-2*radius, width-2*radius)
    
    normals = np.zeros((*image_size, 3))
    
    
    for y in range(height):
        for x in range(width):
            dx = x - center[1]
            dy = -(y - center[0])
            dist_sq = dx**2 + dy**2
            
            if dist_sq <= radius**2:
                if dist_sq == 0:  # Center point
                    normals[y, x] = [0, 0, 1]
                else:
                    dist = np.sqrt(dist_sq)
                    z = np.sqrt(radius**2 - dist_sq)
                    normals[y, x] = [dx/radius, dy/radius, z/radius]
    
    return normals
        
    
    



def main():
    dataFormat = 'PNG'
    bitdepth = 16
    gamma = 1
    resize = (512,612) #(height,width)
    dataNameStack = ['Bear', 'Cat', 'Pot', 'Buddha']

    # Generate sphere for normal reference
    
    sphere_normal = sphere_normals(resize, 50) 

    for testId in range(4):
        dataName = f"{dataNameStack[testId]}{dataFormat}"
        datadir = os.path.join('../pmsData', dataName)
        data = load_datadir_re(datadir, resize, gamma)

        # 1.2.1  Least Squares-Based Method
        Normal = myPMS(data, resize) + sphere_normal
        save_results(Normal, dataNameStack, testId, "Least Squares-Based", resize)

        # 1.2.2  Robust Photometric Stereo
        ratio = 0.2
        Normal = myRobustPMS(data, resize, ratio) + sphere_normal
        save_results(Normal, dataNameStack, testId,
            f"ratio {ratio} Robust Photometric Stereo", resize)
        
        
        # 1.2.2 PCA-Based Method
        rank = 3 # rank <= 96
        Normal = myPCA(data, resize, rank) + sphere_normal
        save_results(Normal, dataNameStack, testId, f"rank {rank} PCA-Based", resize)


        # # 1.2.3 Calculate the depth map and convert it into mesh
        Normal = myPCA(data, resize, 3) # This is arbitrary method.
                                        # You can choose anyone you want.
        X, Y, Surface = mynormal2depth(Normal, testId, dataNameStack)

        # 1.2.3 Convert PC to mesh and save results mesh in standard mesh format
        surface2mesh(X, Y, Surface, testId)
    


if __name__ == "__main__":
    main()