from pathlib import Path
import numpy as np
import pandas as pd

def Color_map(color_map_path: Path):
  '''
    Returns the reversed String.
    Parameters:
        dataframe: A Dataframe with rgb values with class maps.
    Returns:
        code2id: A dictionary with color as keys and class id as values.   
        id2code: A dictionary with class id as keys and color as values.
        name2id: A dictionary with class name as keys and class id as values.
        id2name: A dictionary with class id as keys and class name as values.
  '''
  cls = pd.read_csv(color_map_path)
  color_code = [tuple(cls.drop("name",axis=1).loc[idx]) for idx in range(len(cls.name))]
  code2id = {v: k for k, v in enumerate(list(color_code))}
  id2code = {k: v for k, v in enumerate(list(color_code))}

  color_name = [cls['name'][idx] for idx in range(len(cls.name))]
  name2id = {v: k for k, v in enumerate(list(color_name))}
  id2name = {k: v for k, v in enumerate(list(color_name))}  
  return code2id, id2code, name2id, id2name

def rgb_to_mask(img, color_map):
    ''' 
        Converts a RGB image mask of shape to Binary Mask of shape [batch_size, classes, h, w]
        Parameters:
            img: A RGB img mask
            color_map: Dictionary representing color mappings
        returns:
            out: A Binary Mask of shape [batch_size, classes, h, w]
    '''
    num_classes = len(color_map)
    shape = img.shape[:2]+(num_classes,)
    out = np.zeros(shape, dtype=np.float64)
    for i, cls in enumerate(color_map):
        out[:,:,i] = np.all(np.array(img).reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
    return out.transpose(2,0,1)

def mask_to_rgb(mask, color_map):
    ''' 
        Converts a Binary Mask of shape to RGB image mask of shape [batch_size, h, w, 3]
        Parameters:
            img: A Binary mask
            color_map: Dictionary representing color mappings
        returns:
            out: A RGB mask of shape [batch_size, h, w, 3]
    '''
    single_layer = np.argmax(mask, axis=1)
    output = np.zeros((mask.shape[0],mask.shape[2],mask.shape[3],3))
    for k in color_map.keys():
        output[single_layer==k] = color_map[k]
    return np.uint8(output)