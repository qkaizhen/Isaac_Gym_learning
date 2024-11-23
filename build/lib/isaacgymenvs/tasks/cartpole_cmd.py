# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# demo 1

import numpy as np
import os
import torch
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

class Cartpole_Cmd(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 300 #每回合次数

        self.cfg["env"]["numObservations"] = 5 #自由度 增加控制量
        self.cfg["env"]["numActions"] = 1 #控制输出

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) #从仿真器获取传感器信息 相对URDF-----------------------
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)#转换为torch张量
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]#要用于Tensor维度的重构，即返回一个有相同数据但不同维度的Tens
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)#获取全局参数
        self.root_state = gymtorch.wrap_tensor(root_state_tensor)#转换为torch张量
        self.root_positions = self.root_state[:, 0:3]#依据顺序提取base信息
        self.root_orientations = self.root_state[:, 3:7]
        self.root_linvels = self.root_state[:, 7:10]
        self.root_angvels = self.root_state[:, 10:13]
 
        self.command_pos_range = self.cfg["env"]["randomCommandPosRanges"]#控制指令
        self.commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)#增加控制指令

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):#创建机器人和厂家
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, 0.5 *-spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, 0.5 *spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf" #载入模型

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True #固定基座 估计基座没有base root无效
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)#从模型获取自由度

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []
        self.env_root=[]
        for i in range(self.num_envs):#并形构建虚拟空间
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT #采用扭矩模式
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]
        cart_vel = self.obs_buf[:, 1]
        cart_pos = self.obs_buf[:, 0]
        command  = self.obs_buf[:, 4]
        #print("command",command) 
        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            command,
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim) #从仿真器获取数据

        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()#小车位置 相对还是绝对
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()#小车速度
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()#倒立摆角度
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()#倒立摆角速度
        self.obs_buf[env_ids, 4] = self.commands[env_ids].squeeze()
    
        #获取全局信息
        self.root_positions = self.root_state[:, 0:3]
        self.root_orientations = self.root_state[:, 3:7]
        self.root_linvels = self.root_state[:, 7:10]
        self.root_angvels = self.root_state[:, 10:13]
        #print(self.root_posbase_positions[env_ids].squeeze())
               # for i in range(self.num_envs):
        base_pos = (self.root_state[env_ids, :3]).cpu().numpy()
        #print(base_pos)
        return self.obs_buf

    def reset_idx(self, env_ids):#重置本id的环境直接通过设置仿真器接口
        positions = 1 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5) #随机设置状态位置
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5) #随机设置状态速度

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.commands[env_ids] = torch_rand_float(-self.command_pos_range , self.command_pos_range, (len(env_ids), 1), device=self.device)#随机控制指令
        #print("command",env_ids,self.commands[env_ids])
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):#计算控制输出
        # if self.progress_buf[0] >= 300:
        #     self.commands[:] = torch_rand_float(-self.command_pos_range , self.command_pos_range, (self.num_envs, 1), device=self.device)#随机控制指令  
        #     print(self.commands)
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor) #将torch数据还原为gym调用
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1
  
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        # debug viz
        self.gym.clear_lines(self.viewer)
        if 1:#self.viewer and self.debug_viz:
            for i in range(self.num_envs):
                #i=23
                #self.cfg["env"]['envSpacing']
                origin = self.gym.get_env_origin(self.envs[i])#<<-----------------获取空间原点
                location=(origin.x, self.commands[i]+origin.y , 2.0)
                color=(1, 1, 0)
                self.draw_sphere(location,color)
                #gymutil.draw_lines(sphere_geom, self.gym, self.viewer, 0, sphere_pose)

    def draw_sphere(self, location, color):
        #self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 3, 3, None, color)
        pose = gymapi.Transform(gymapi.Vec3(location[0], location[1], location[2]), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], pose)

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_cartpole_reward(command,pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle*0.5 - 0.01 * torch.abs(cart_vel) - 0.01 * torch.abs(pole_vel) - torch.abs(command-cart_pos)*0.8
    #print(torch.abs(command-cart_pos))
    #计算奖励 倒立摆偏离大奖励小  速度越大奖励小  

    # adjust reward for reset agents 复位奖励
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)#小车位置超出偏差给负数奖励
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)#倒立摆摔倒 给负奖励

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf) #当达到前面条件 reset_buf进行赋值 
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset
 