import numpy as np
import torch
import config

def move_data_to_gpu(x, cuda):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x
    
    
def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
    
    
def forward(model, generate_func, cuda, return_input=False, 
    return_target=False):
    '''Forward data to model in mini-batch. 
    
    Args: 
      model: object
      generate_func: function
      cuda: bool
      return_input: bool
      return_target: bool
      max_validate_num: None | int, maximum mini-batch to forward to speed up validation
    '''
    output_dict = {}
    audio_num = config.audio_num
    # Evaluate on mini-batch
    for batch_data_dict in generate_func:
        
        start = batch_data_dict['start']
        end = batch_data_dict['end']
        for j in range(batch_data_dict['target'].shape[0]):
            for i in range(audio_num):
                if (start[j]>=float(i) and end[j]<=float(i+3)) or (start[j]<=float(i+3) and end[j]>=float(i+3) and start[j]<=float(i+3)-0.5) or (start[j]<=float(i) and end[j]>=float(i) and end[j]>=float(i)+0.5) or ((start[j]<=float(i) and end[j]>=float(i+3))):
                    x = np.array([[1.,0.]])
                    gt = x
                else :
                    x = np.array([[0.,1.]])
                    gt = x
                with torch.no_grad():
                    model.eval()
                    batch_feature = move_data_to_gpu(batch_data_dict['feature'][j, i, :, :], cuda)
                    batch_feature = batch_feature[None, :, :]
                    batch_output = model(batch_feature)
                append_to_dict(output_dict, 'output', batch_output.data.cpu().numpy())
                append_to_dict(output_dict, 'target', gt)    
                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict
