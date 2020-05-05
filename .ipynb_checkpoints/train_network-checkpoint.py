import torch

'''
    Takes a state or batch of states and converts
    them into a pytorch tensor format. Shapes should
    be (batch_size x 1 x 4 x 4) or (1 x 1 x 4 x 4)
'''
def process_state(state):
    state = torch.Tensor(state)
    if (state.shape == torch.Size([4,4])):
        desired_shape = (1, 1, *state.shape)
        state = state.view(*desired_shape)
    else:
        desired_shape = (state.shape[0], 1, state.shape[1], state.shape[2])
        state = state.view(*desired_shape)
    return state