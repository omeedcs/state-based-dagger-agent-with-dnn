import torch
import numpy as np

from .model import Model
from .utils import load_recording
from .player import extract_featuresV2

# from .models import Detector, save_model
# from .utils import load_detection_data
# from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    epochs = 500
    
    # set up the optimizer similar to what we did in class
    optimizer = torch.optim.Adam(
      model.parameters(), 
      lr = 1e-4,
      weight_decay = 1e-5)

    # set up the training data based on the given function. We were given this
    # path based on what the teacher said in the piazza post.

    collectFromTeam1 = ['jurgen_vs_ai',
    'jurgen_vs_geoffrey',
    'jurgen_vs_jurgen',
    'jurgen_vs_image_jurgen',
    'jurgen_vs_state',
    'jurgen_vs_yann',
    'jurgen_vs_yoshua']

    collectFromTeam2 = ['ai_vs_jurgen',
    'geoffrey_vs_jurgen',
    'jurgen_vs_jurgen',
    'image_jurgen_vs_jurgen',
    'state_vs_jurgen',
    'yann_vs_jurgen',
    'yoshua_vs_jurgen']

    data_train = []

    for gd in collectFromTeam1:
      game_data = load_recording(gd)

      for row in game_data:
        features_team1_p0 = extract_featuresV2(row['team1_state'][0], 
        row['soccer_state'], 
        row['team2_state'], 
        row['team1_state'][0]['kart']['player_id'] % 2)

        features_team1_p2 = extract_featuresV2(row['team1_state'][1], 
        row['soccer_state'], 
        row['team2_state'], 
        row['team1_state'][1]['kart']['player_id'] % 2)

        # print(row['actions'])
        # construct a (state, action) tuple for data_train.
        params = ['acceleration', 'steer', 'brake']
        for i in range(0, 4):
          for param in params:
            if (param not in row['actions'][i]):
              row['actions'][i][param] = 0
            if param == 'steer':
              row['actions'][i][param] = (row['actions'][i][param]/2) + (1/2)

        action_tup_0 = (row['actions'][0]['acceleration'], row['actions'][0]['steer'], row['actions'][0]['brake'])
        tup_0 = (features_team1_p0, torch.tensor(action_tup_0))
        action_tup_2 = (row['actions'][2]['acceleration'], row['actions'][2]['steer'], row['actions'][2]['brake'])
        tup_2 = (features_team1_p2, torch.tensor(action_tup_2))

        data_train.append(tup_0)
        data_train.append(tup_2)

    for gd in collectFromTeam2:
      game_data = load_recording(gd)
      for row in game_data:


        features_team2_p1 = extract_featuresV2(row['team2_state'][0], 
        row['soccer_state'],
        row['team1_state'], 
        row['team2_state'][0]['kart']['player_id'] % 2)

        features_team2_p3 = extract_featuresV2(row['team2_state'][1], 
        row['soccer_state'],
        row['team1_state'], 
        row['team2_state'][1]['kart']['player_id'] % 2)
        
        params = ['acceleration', 'steer', 'brake']
        for i in range(0, 4):
          for param in params:
            if (param not in row['actions'][i]):
              row['actions'][i][param] = 0
            if param == 'steer':
              row['actions'][i][param] = (row['actions'][i][param]/2) + (1/2)

        action_tup_1 = (row['actions'][1]['acceleration'], row['actions'][1]['steer'], row['actions'][1]['brake'])
        tup_1 = (features_team2_p1, torch.tensor(action_tup_1))
        action_tup_3 = (row['actions'][3]['acceleration'], row['actions'][3]['steer'], row['actions'][3]['brake'])
        tup_3 = (features_team2_p3, torch.tensor(action_tup_3))


        data_train.append(tup_1)
        data_train.append(tup_3)



    # use the loss function that we created.
    loss = torch.nn.SmoothL1Loss(beta = .2)

    # iterate through the epochs.
    for epoch in range(epochs):

      # display the current epoch.
      print("Current Epoch: ", epoch)

      # train our model.
      model.train()

      # go through our training data.

      epoch_loss = 0
      for x, y in data_train:
        
        # similar logic to course work.
        x = x.to(device)
        y = y.to(device)

        # use the functions we created.
        output = model(x)

        #output[1] = (output[1] * 2) - 1
        
        # compute the loss
        cur_loss = loss(output, y).mean()

        epoch_loss += cur_loss

        # zero out the gradient.
        optimizer.zero_grad()

        # back propogate
        cur_loss.backward()

        # step forward
        optimizer.step()
      
      # # save the model on the current iteration.
      # save_model(model)
      print (epoch_loss)
      # save the model on the next iteration.
      # save_model(model)
      script = torch.jit.script(model)
      torch.jit.save(script, 'goated_agent/goated_agent.pt')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
