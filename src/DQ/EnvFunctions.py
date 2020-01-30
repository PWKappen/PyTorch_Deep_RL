import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

resize = T.Compose([T.ToPILImage(),
                    T.Scale((84,84), interpolation=Image.NEAREST),
                    T.ToTensor()])

screen_width = 600

def get_cart_location(env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen(env):
    screen = np.expand_dims(np.mean(env.render(mode='rgb_array'),axis=2).transpose((0,1)),axis=0)
    screen = screen[:, 160:320]
    # view_width = 320
    # cart_location = get_cart_location(env)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width //2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width //2,
    #                         cart_location + view_width //2)
    # screen = screen[:,:,slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)/255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).type(Tensor)

def adapt_image(image):
    screen = np.expand_dims(np.mean(image,axis=2),axis=0)
    # view_width = 320
    # cart_location = get_cart_location(env)
    # if cart_location < view_width // 2:
    #     slice_range = slice(view_width)
    # elif cart_location > (screen_width - view_width //2):
    #     slice_range = slice(-view_width, None)
    # else:
    #     slice_range = slice(cart_location - view_width //2,
    #                         cart_location + view_width //2)
    # screen = screen[:,:,slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32)
    screen = torch.from_numpy(screen).type(torch.ByteTensor)
    screen = resize(screen).unsqueeze(0)*255
    screen = screen.type(Tensor)
    return screen
