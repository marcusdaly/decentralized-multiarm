from torch.utils.data import Dataset
from torch import tensor, cat, stack, FloatTensor, Tensor, save
import ray
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from .BaseNet import StochasticActor
from torch.optim import Adam
from torch.nn.functional import mse_loss
from time import time
from tensorboardX import SummaryWriter
from os import mkdir
from os.path import exists, abspath
import numpy as np
import pybullet as p
from itertools import chain



workspace_radius = 0.85

class PolicyManager:
    def __init__(self):
        self.policies = {}
        self.memory_map = {}

    def register_policy(self, name, entry):
        self.policies[name] = entry
        self.memory_map[name] = ray.get(
            entry.get_memory_cluster.remote())['handle']
        ray.get(entry.set_handle.remote({'handle': entry}))

    def get_inference_node(self, key):
        return ray.get(self.policies[
            key].get_inference_node.remote())

    def get_inference_nodes(self):
        return dict([(p,
                      self.get_inference_node(p))
                     for p in self.policies])


class ReplayBufferDataset(Dataset):
    def __init__(self, data, device, capacity):
        self.keys = list(data)
        self.data = data
        self.device = device
        self.capacity = int(capacity)
        self.observations_padded = None
        self.next_observations_padded = None
        self.freshness = 0.0

    def __len__(self):
        return len(self.data['observations'])

    def extend(self, data):
        len_new_data = len(data['observations'])
        for key in self.data:
            self.data[key].extend(data[key])
        if len(self) != 0:
            self.freshness += len_new_data / len(self)
        else:
            self.freshness = 0.0

        if len(self.data['observations']) > self.capacity:
            amount = len(self.data['observations']) - self.capacity
            for key in self.data:
                del self.data[key][:amount]

        self.padded_observations = None
        self.next_observations_padded = None

    def pad_observations(self, input):
        # print(input)
        flat_obs= [ob[0] for ob in input]
        img_obs= [ob[1] for ob in input]
        flat_obs = pad_sequence(flat_obs, batch_first=True)
        img_obs = pad_sequence(img_obs, batch_first=True)
        observations = [(flat_obs[i], img_obs[i]) for i in range(flat_obs.shape[0])]
        return observations

    def __getitem__(self, idx):
        if self.observations_padded is None or\
                self.next_observations_padded is None:
            self.observations_padded = self.pad_observations(
                self.data['observations'])
            self.next_observations_padded = self.pad_observations(
                self.data['next_observations'])
            if 'critic_observations' in self.data\
                    and len(self.data['critic_observations']) > 0:
                self.critic_observations_padded = self.pad_observations(
                    self.data['critic_observations'])
                self.critic_next_observations_padded = self.pad_observations(
                    self.data['critic_next_observations'])
            else:
                self.critic_observations_padded = \
                    self.observations_padded
                self.critic_next_observations_padded =\
                    self.next_observations_padded
        if 'critic_observations' in self.data:
            return (
                (self.critic_observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.critic_observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                (self.observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                (self.critic_next_observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.critic_next_observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                (self.next_observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.next_observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )
        else:
            return (
                (self.observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                self.data['actions'][idx].to(
                    self.device, non_blocking=True),
                tensor(float(self.data['rewards'][idx])).to(
                    self.device, non_blocking=True),
                (self.next_observations_padded[idx][0].to(
                    self.device, non_blocking=True), self.next_observations_padded[idx][1].to(
                    self.device, non_blocking=True)),
                tensor(float(self.data['is_terminal'][idx])).to(
                    self.device, non_blocking=True)
            )


def global_to_ur5_frame(base_pos, position, rotation=None):
    self_pos, self_rot = base_pos
    invert_self_pos, invert_self_rot = p.invertTransform(
        self_pos, self_rot)
    ur5_frame_pos, ur5_frame_rot = p.multiplyTransforms(
        invert_self_pos, invert_self_rot,
        position, invert_self_rot if rotation is None else rotation
    )
    return ur5_frame_pos, ur5_frame_rot


# TODO may want to actual use config to load this......
def preprocess_experiences(experiences):
    obs_key = [
        "joint_values",
        "end_effector_pose",
        "target_pose",
        "link_positions",
        "pose",
    ]
    histories = [
        1,
        1,
        1,
        1,
        0,
    ]

    history = []
    states = []

    # first, add in history to the state.
    for exp in experiences:
        state = exp[0]
        if len(history) == 0:
            history.append(state)


        states.append([])
        for ur5_state, hist_ur5_state in zip(state['ur5s'], history[-1]['ur5s']):
            states[-1].append({})

            for key, num_hist in zip(obs_key, histories):
                val = None
                if key == 'joint_values':
                    val = [curr_ur5_state[key]
                           for curr_ur5_state in ([hist_ur5_state, ur5_state] if num_hist > 0 else [ur5_state])]
                elif 'link_positions' in key:
                    # get flatten link positions in ur5's frame of reference
                    val = [list(chain.from_iterable(
                        [
                            global_to_ur5_frame(
                            base_pos=curr_ur5_state["pose"],
                            position=np.array(link_pos),
                            rotation=None)[0]
                         for link_pos in curr_ur5_state[key]]))
                        for curr_ur5_state in ([hist_ur5_state, ur5_state] if num_hist > 0 else [ur5_state])]
                elif 'end_effector_pose' in key or \
                        'target_pose' in key\
                        or key == 'pose' or key == 'pose_high_freq':
                    val = [list(chain.from_iterable(
                        global_to_ur5_frame(
                            base_pos=curr_ur5_state["pose"],
                            position=curr_ur5_state[key][0],
                            rotation=curr_ur5_state[key][1])))
                           for curr_ur5_state in ([hist_ur5_state, ur5_state] if num_hist > 0 else [ur5_state])]
                else:
                    val = [
                        global_to_ur5_frame(
                        base_pos=curr_ur5_state["pose"],
                        position=curr_ur5_state[key])
                        for curr_ur5_state in ([hist_ur5_state, ur5_state] if num_hist > 0 else [ur5_state])]
                states[-1][-1][key] = val

            states[-1][-1]["image"] = ur5_state["image"]

        history.append(state)
        del history[0]

    # print(states[0])

    all_ur5_observations = []

    # for each state...
    for state in states:
        state_observations = []
        # for each ur5 observation....
        for this_ur5_idx, _ in enumerate(state):
            pos = np.array(state[this_ur5_idx]["pose"])

            # sort according to difference in pose from this ur5
            # print(state[0]["pose"])
            # print(pos)
            sorted_ur5s = [ur5 for ur5 in state
                           if np.linalg.norm(
                               pos - np.array(ur5["pose"]))
                           < 2 * workspace_radius
            ]
            # Sort by base distance, furthest to closest
            sorted_ur5s.sort(reverse=True, key=lambda ur5:
                             np.linalg.norm(pos - np.array(ur5["pose"])))

            state_observations.append({"flat": sorted_ur5s, "image": state[this_ur5_idx]["image"]})

        all_ur5_observations.append(state_observations)

    outputs = []
    img_outputs = []
    for experience_obs in all_ur5_observations:
        experience_outputs = []
        experience_img_outputs = []
        for obs in experience_obs:
            flat_obs = obs["flat"]
            output = []
            for ur5_obs in flat_obs:
                ur5_output = np.array([])
                for key in obs_key:
                    item = ur5_obs[key]
                    for history_frame in item:
                        ur5_output = np.concatenate((
                            ur5_output,
                            history_frame))
                output.append(ur5_output)

            # 107 dim now
            output = torch.FloatTensor(np.array(output))

            # 3x64x64
            img_output = obs["image"]

            # channels before h/w
            img_output = torch.permute(torch.FloatTensor(img_output), (2, 0, 1))

            experience_outputs.append(output)
            experience_img_outputs.append(img_output)

        outputs.append(experience_outputs)
        img_outputs.append(experience_img_outputs)

    return outputs, img_outputs

class BehaviourCloneDataset(Dataset):
    def __init__(self, path):
        self.observations = []
        self.flat_observations = []
        self.img_observations = []
        self.actions = []
        for file_name in tqdm(
                list(Path(path).rglob('*.pt')),
                desc='importing trajectories'):
            with open(file_name, 'rb') as file:
                experiences = torch.load(file)
                flat_obs, img_obs = preprocess_experiences(experiences)
                flat_obs = chain.from_iterable(flat_obs)
                img_obs = chain.from_iterable(img_obs)
                actions = chain.from_iterable([list(exp[1]) for exp in experiences])
                self.flat_observations.extend(flat_obs)
                self.img_observations.extend(img_obs)
                self.actions.extend(actions)

        self.flat_observations = pad_sequence(
            self.flat_observations,
            batch_first=True)
        self.img_observations = torch.stack(
            self.img_observations,
            dim=0)
        self.actions = torch.stack(self.actions, dim=0)

    def __len__(self):
        return len(self.flat_observations)

    def __getitem__(self, idx):
        return ((self.flat_observations[idx], self.img_observations[idx]), self.actions[idx])


def setup_behaviour_clone(args, config, obs_dim, obs_img_width, img_encoding_dim, device):
    dataset = BehaviourCloneDataset(args.expert_trajectories)
    train_loader = DataLoader(
        dataset,
        batch_size=config['behaviour-clone']['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=0)
    policy = StochasticActor(
        obs_dim=obs_dim,
        action_dim=6,
        obs_img_width=obs_img_width,
        img_encoding_dim=img_encoding_dim,
        action_variance_bounds=config['training']['action_variance'],
        network_config=config['training']['network']['actor']).to(device)
    optimizer = Adam(
        policy.parameters(),
        lr=config['behaviour-clone']['lr'],
        weight_decay=config['behaviour-clone']['weight_decay'],
        betas=(0.9, 0.999))
    print(policy)
    print(optimizer)

    def train():
        batch_count = 0
        logdir = "runs/" + args.name
        if not exists(logdir):
            mkdir(logdir)
        writer = SummaryWriter(logdir)
        for epoch in range(config['behaviour-clone']['epochs']):
            pbar = tqdm(train_loader)
            epoch_loss = 0.0
            start = time()
            for batch_idx, ((flat_observations, img_observations), expert_actions) in enumerate(pbar):
                policy_action_dist = policy(
                    (flat_observations.to(device, non_blocking=True), img_observations.to(device, non_blocking=True)),
                    deterministic=False,
                    reparametrize=True,
                    return_dist=True)
                loss = - \
                    policy_action_dist.log_prob(
                        expert_actions.to(device, non_blocking=True)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1
                writer.add_scalar(
                    'Training/Policy_Loss',
                    loss.item(),
                    batch_count)
                epoch_loss += loss.item()
                pbar.set_description(
                    'Epoch {} | Batch {} | Loss: {:.05f}'.format(
                        epoch, batch_idx,
                        loss.item()))
            pbar.set_description('Epoch {} | Loss: {:.05f} | {:.01f} seconds'.format(
                epoch, epoch_loss / (batch_idx + 1),
                float(time() - start)))
            output_path = abspath("{}/ckpt_{:05d}".format(
                writer.logdir,
                epoch))
            save({
                'networks': {
                    'policy': policy.state_dict(),
                    'opt': optimizer.state_dict()
                },
                'stats': {
                    'batches': batch_count,
                    'epochs': epoch
                }
            }, output_path)
            print("Saved checkpoint at " + output_path)
    return train
