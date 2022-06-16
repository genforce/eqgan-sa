# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from re import T
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import torchvision.models as models
import torch.nn.functional as F

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------
try:
    nan_to_num = torch.nan_to_num # pytorch > 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


# To store gradients and intermediate features

grad_list  = list()
act_list = list()

def grad_hook(module, grad_input, grad_output):
    if grad_output[0] is None:
        return None
    grad_list.append(grad_output[0].detach())

def activation_hook(module, input, output):
    act_list.append(output[0])



class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, align_loss=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.align_loss = align_loss



        # register hook to b8, b16, and b32 modules
        # here is a bit complicated because 
        # the functions may be wrapped by .module or not

        # and different pytorch versions use different names for hook functions,
        # e.g., register_full_backward_hook or register_backward_hook
        
        # hook b8
        try:
            target_layer = self.D.module.b8     # if wrapped by .module
        except:
            target_layer = self.D.b8            # if not wrapped by .module

        try:
            # if the pytorch version supports register_full_backward_hook
            target_layer.register_full_backward_hook(grad_hook)
        except:
            # if the pytorch version supports register_backward_hook
            target_layer.register_backward_hook(grad_hook)

        target_layer.register_forward_hook(activation_hook)

        # hook b16 and b32
        try:
            # if wrapped by .module
            self.D.module.b16.register_forward_hook(activation_hook)
            self.D.module.b32.register_forward_hook(activation_hook)
            try:
                # if the pytorch version supports register_full_backward_hook
                self.D.module.b16.register_full_backward_hook(grad_hook)
                self.D.module.b32.register_full_backward_hook(grad_hook)
            except:
                # if the pytorch version supports register_backward_hook
                self.D.module.b16.register_backward_hook(grad_hook)
                self.D.module.b32.register_backward_hook(grad_hook) 
        except:
            # if not wrapped by .module
            self.D.b16.register_forward_hook(activation_hook)
            self.D.b32.register_forward_hook(activation_hook)
            try:
                # if the pytorch version supports register_full_backward_hook
                self.D.b16.register_full_backward_hook(grad_hook)
                self.D.b32.register_full_backward_hook(grad_hook)
            except:
                # if the pytorch version supports register_backward_hook
                self.D.b16.register_backward_hook(grad_hook)
                self.D.b32.register_backward_hook(grad_hook)
                  
        self.align_thres = 0.25

    def run_G(self, z, c, sync, heatmap=None,return_hm_feat=False):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        # return both the generated image and the randomly constructed heatmap
        with misc.ddp_sync(self.G_synthesis, sync):
            img, hm = self.G_synthesis(ws,return_hm=True,heatmap=heatmap,return_hm_feat=return_hm_feat)

        return img, ws, hm

    def run_D(self, img, c, sync, return_img=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)

        with misc.ddp_sync(self.D, sync):
            # record_inter: also save the intermediate feat
            logits, inter_feat = self.D(img, c, record_inter=True)

        if return_img:
            return logits, inter_feat, img
        return logits, inter_feat


    def compute_GradCam_multi(self, img, only_weight=False, return_list = False, to_eval=True):
        # compute GradCAM at multiple levels

        # lists to save intermediate features and gradients
        global grad_list; global act_list
        grad_list = list(); act_list = list()

        # detach the gradients of the input image
        # so the operation of computing GradCAM would not generator
        img = img.detach().clone()
        img = img.requires_grad_(True)

        if to_eval: self.D = self.D.eval()

        logits = self.D(img)

        loss_Dgen=0
        for i in range(len(logits)):    # using this for loop to preserve the batch dim
            loss_Dgen = loss_Dgen + torch.nn.functional.softplus(-logits[i])

        loss_Dgen.backward()
        weight_list = {}

        # convert gradients to GradCAM weights
        for num in range(len(grad_list)):
            grads = grad_list[num].detach()
            weights = torch.mean(grads,dim=(2,3),keepdim=True)
            res = grads.shape[-1]*2
            weight_list[f'b{res}'] = weights.detach().clone().contiguous()
        
        grad_list = list(); act_list = list()
        img = img.requires_grad_(False)

        if to_eval: self.D = self.D.train()


        if return_list:
            return weight_list
        elif only_weight:
            return weights.detach().clone().contiguous()
        else:
            return None



    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cur_nimg=None):

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)



        # Gmain: Maximize logits for generated images.
        global grad_list
        global act_list

        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_heatmap = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.

                training_stats.report('Stat/gen/mean', gen_img.mean().detach())
                training_stats.report('Stat/gen/max', gen_img.max().detach())
                training_stats.report('Stat/gen/std', gen_img.std().detach())

                gen_logits, inter_feat, aug_img = self.run_D(gen_img, gen_c, sync=False, return_img=True)
                
                training_stats.report('Scores/fake_med', gen_logits.median())
                training_stats.report('Scores/fake_max', gen_logits.max())
                training_stats.report('Scores/fake_std', gen_logits.std())
                training_stats.report('Scores/fake', gen_logits)
                training_stats.report('Signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                loss_Gmain = loss_Gmain.mean()

                if self.align_loss:
                    # compute GradCAM weights at various levels
                    gradcam_weights = self.compute_GradCam_multi(aug_img,only_weight=True,return_list=True)
                    
                    norm_act_dict = {}
                    lw_dict = {'b8':1.0,'b16':1.0,'b32':1.0}        # loss weights for various levels

                    align_loss_sum = 0
                    resolution = aug_img.shape[-1]

                    for key_name in gradcam_weights:                # each level

                        # weights * features
                        act_maps = gradcam_weights[key_name] * inter_feat[key_name]
                        # activation
                        act_maps_relu = F.relu(act_maps).sum(dim=1,keepdim=True)

                        cur_gen_heatmap = gen_heatmap[key_name]

                        if cur_gen_heatmap.shape[1]>1:
                            assert cur_gen_heatmap.shape[1]<16      # check size

                            # normalize heatmap range
                            cur_gen_heatmap = cur_gen_heatmap.mean(dim=1,keepdim=True)

                            cur_gen_heatmap_min = cur_gen_heatmap.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
                            norm_cur_gen_heatmap = cur_gen_heatmap-cur_gen_heatmap_min

                            norm_cur_gen_heatmap_max = norm_cur_gen_heatmap.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
                            cur_gen_heatmap = norm_cur_gen_heatmap/norm_cur_gen_heatmap_max          

                        # upsample to the resolution
                        acts = F.interpolate(act_maps_relu,(resolution,resolution),mode='bilinear', align_corners=True)

                        # normalize the acts maps
                        # this step can be skipped
                        # we keep it to avoid numerical errors
                        acts_min = acts.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
                        norm_acts = acts-acts_min

                        acts_max = norm_acts.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1).detach()
                        norm_acts = norm_acts/acts_max

                        norm_acts = nan_to_num(norm_acts, posinf=1.0)

                        norm_act_dict[key_name] = norm_acts
                        
                        # alignment loss, simple L1 distance here
                        alignment_loss = (norm_acts-cur_gen_heatmap).abs()

                        nan_flags = torch.isnan(alignment_loss) + torch.isinf(alignment_loss)                        
                        alignment_loss = alignment_loss[~nan_flags]

                        training_stats.report('Stat/align/nan_num_'+key_name, nan_flags.sum())
                        training_stats.report('Stat/align/hm_mean_'+key_name, cur_gen_heatmap.mean().item())
                        training_stats.report('Stat/align/hm_std_'+key_name, cur_gen_heatmap.std().item())

                        training_stats.report('Stat/align/act_mean_'+key_name, norm_acts.mean().item())
                        training_stats.report('Stat/align/act_std_'+key_name, norm_acts.std().item())  

                        if self.align_thres is not None:
                            if len(alignment_loss[alignment_loss>self.align_thres])>1:
                                # if has valid data points
                                alignment_loss =  alignment_loss[alignment_loss>self.align_thres].mean() 
                            else:
                                alignment_loss = alignment_loss.mean() 

                        if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                            print('WARNINGGGGGG! alignment_loss NAN!!!!!!!!!!!!!')
                        align_loss_sum = align_loss_sum + alignment_loss*lw_dict[key_name]
                    
                    
                    alignment_loss = align_loss_sum * self.alignment_loss
                 
                training_stats.report('Loss/G/align_loss', alignment_loss)

                loss_Gmain = loss_Gmain + alignment_loss

            with torch.autograd.profiler.record_function('Gmain_backward'):
                grad_list = list();act_list = list()      # empty hook 
                loss_Gmain.mul(gain).backward()


        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink

                gen_img, gen_ws, gen_heatmap = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync, return_hm_feat=True)


                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]


                if isinstance(pl_grads, tuple):
                    ws_pl_lengths = pl_grads[0].square().sum(2).mean(1).sqrt()
                    hm_pl_lengths = pl_grads[1].square().sum(3).sum(2).mean(1).sqrt()

                    pl_lengths = ws_pl_lengths + hm_pl_lengths
                    training_stats.report('PL/ws_pl_lengths', ws_pl_lengths.mean())
                    training_stats.report('PL/hm_pl_lengths', hm_pl_lengths.mean())
                else:
                    pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()

                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)

                self.pl_mean.copy_(pl_mean.detach())

                pl_penalty = (pl_lengths - pl_mean).square()
                
                training_stats.report('Loss/pl_penalty', pl_penalty.mean())
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl.mean())
            with torch.autograd.profiler.record_function('Gpl_backward'):
                grad_list = list();act_list = list()    # empty hook
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_heatmap = self.run_G(gen_z, gen_c, sync=False)
                gen_logits, _ = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.

                training_stats.report('Scores/fake_med', gen_logits.median())
                training_stats.report('Scores/fake_max', gen_logits.max())
                training_stats.report('Scores/fake_std', gen_logits.std())

                training_stats.report('Scores/fake', gen_logits)
                training_stats.report('Signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                grad_list = list(); act_list=list()
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)

                real_logits, _ = self.run_D(real_img_tmp, real_c, sync=sync)

                training_stats.report('Scores/real_med', real_logits.median())
                training_stats.report('Scores/real_min', real_logits.min())
                training_stats.report('Scores/real_std', real_logits.std())

                training_stats.report('Scores/real', real_logits)
                training_stats.report('Signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                grad_list = list(); act_list=list()
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
