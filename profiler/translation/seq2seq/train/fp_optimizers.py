# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math

import torch
from torch.nn.utils import clip_grad_norm_

from seq2seq.utils import fused_norm
from apex.optimizers import FusedAdam

class Fp16Optimizer:
    """
    Mixed precision optimizer with dynamic loss scaling and backoff.
    https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#scalefactor
    """
    @staticmethod
    def set_grads(params, params_with_grad):
        """
        Copies gradients from param_with_grad to params

        :param params: dst parameters
        :param params_with_grad: src parameters
        """
        for param, param_w_grad in zip(params, params_with_grad):
            if param.grad is None:
                param.grad = torch.nn.Parameter(torch.empty_like(param))
            param.grad.data.copy_(param_w_grad.grad.data)

    @staticmethod
    def set_weights(params, new_params):
        """
        Copies parameters from new_params to params

        :param params: dst parameters
        :param new_params: src parameters
        """
        for param, new_param in zip(params, new_params):
            param.data.copy_(new_param.data)


    # Flattening master weight
    def initialize_flat_fp32_weight(self, model):
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        for p in self.fp16_model.parameters():
            p.grad = None

        nelem = 0
        for p in model.parameters():
            nelem += p.numel()
        self.fp32_params = torch.cuda.FloatTensor(nelem)
        self.fp16_params = torch.cuda.HalfTensor(nelem)

        pointer = 0
        for p in model.parameters():
            nelem = p.numel()
            self.fp32_params[pointer:pointer+nelem].copy_(p.data.view(-1))
            self.fp16_params[pointer:pointer+nelem].copy_(p.data.view(-1))
            pointer += nelem

        self.fp32_params = torch.nn.Parameter(self.fp32_params)
        self.fp32_params.grad = torch.autograd.Variable(
            self.fp32_params.data.new(*self.fp32_params.size()))
        self.fp16_params = torch.nn.Parameter(self.fp16_params)
        self.fp16_params.grad = torch.autograd.Variable(
            self.fp16_params.data.new(*self.fp16_params.size()))

    @staticmethod
    def fp16_to_fp32_flat_grad(fp32_params, fp16_model):
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            fp32_params.grad.data[pointer:pointer+nelem].copy_(p.grad.data.view(-1))
            pointer += nelem

    @staticmethod
    def fp16_to_fp16_flat_grad(fp16_params, fp16_model):
        fp16_params.grad.data = torch.cat(
            [p.grad.data.view(-1) for p in fp16_model.parameters()])

    @staticmethod
    def fp32_to_fp16_grads(fp16_model, fp32_params):
        #Copy master weights onto model weights
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            p.data.view(-1).copy_(fp32_params.data[pointer:pointer+nelem])
            pointer += nelem

    @staticmethod
    def fp16_to_fp16_grads(fp16_model, fp16_params):
        #Copy master weights onto model weights
        pointer = 0
        for p in fp16_model.parameters():
            nelem = p.numel()
            p.data.view(-1).copy_(fp16_params.data[pointer:pointer+nelem])
            pointer += nelem


    def __init__(self, fp16_model, grad_clip=float('inf'), loss_scale=8192,
                 dls_downscale=2, dls_upscale=2, dls_upscale_interval=128):
        logging.info('Initializing fp16 optimizer')
        self.initialize_flat_fp32_weight(fp16_model)
        self.since_last_invalid = 0
        self.loss_scale = loss_scale
        self.dls_downscale = dls_downscale
        self.dls_upscale = dls_upscale
        self.dls_upscale_interval = dls_upscale_interval
        self.grad_clip = grad_clip
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def initialize_model(self, model):
        """
        Initializes internal state and build fp32 master copy of weights.

        :param model: fp16 model
        """
        logging.info('Initializing fp32 clone weights')
        self.fp16_model = model
        for p in self.fp16_model.parameters():
            p.grad = None
        self.fp32_params = [param.to(torch.float32).detach()
                            for param in model.parameters()]

        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.
        Applies loss scaling, computes gradients in fp16, converts gradients to
        fp32, inverts scaling and applies optional gradient norm clipping.
        If gradients are finite, it applies update to fp32 master weights and
        copies updated parameters to fp16 model for the next iteration. If
        gradients are not finite, it skips the batch and adjusts scaling factor
        for the next iteration.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss *= self.loss_scale

        for p in self.fp16_model.parameters():
            p.grad = None
        loss.backward()

        scaling_factor = self.loss_scale
        if isinstance(optimizer, FusedAdam):
            if self.world_size != 1 and self.fp16_model.retain_allreduce_buffers:
                assert len(self.fp16_model.allreduce_buffers) == 1
                self.fp16_params.grad.data = self.fp16_model.allreduce_buffers[0]

                # Average the all-reduced gradients by world size if APEX
                # doesn't do that
                if not self.fp16_model.gradient_average:
                    scaling_factor *= self.world_size
            else:
                self.fp16_to_fp16_flat_grad(self.fp16_params, self.fp16_model)

            norm = fused_norm(self.fp16_params.grad.data) / scaling_factor
        else:
            self.fp16_to_fp32_flat_grad(self.fp32_params, self.fp16_model)
            if scaling_factor != 1.0:
                self.fp32_params.grad.data /= scaling_factor

            norm = clip_grad_norm_([self.fp32_params], self.grad_clip)

        if update:
            if math.isfinite(norm):
                scheduler.step()
                if isinstance(optimizer, FusedAdam):
                    clip_coef = self.grad_clip / (norm + 1e-6)
                    if clip_coef >= 1:
                        clip_coef = scaling_factor
                    else:
                        clip_coef = scaling_factor / clip_coef
                    optimizer.step(grads=[self.fp16_params.grad], scale=clip_coef)
                else:
                    optimizer.step()
                self.fp32_to_fp16_grads(self.fp16_model, self.fp32_params)
                self.since_last_invalid += 1
            else:
                self.loss_scale /= self.dls_downscale
                self.since_last_invalid = 0
                logging.info(f'Gradient norm: {norm}')
                logging.info(f'Skipped batch, new scale: {self.loss_scale}')

            if self.since_last_invalid >= self.dls_upscale_interval:
                self.loss_scale *= self.dls_upscale
                self.loss_scale = min(self.loss_scale, 8192.0)
                logging.info(f'Upscaling, new scale: {self.loss_scale}')
                self.since_last_invalid = 0


class Fp32Optimizer:
    """
    Standard optimizer, computes backward and applies weight update.
    """
    def __init__(self, model, grad_clip=None):
        """
        Constructor for the Fp32Optimizer

        :param model: model
        :param grad_clip: max value of gradient norm
        """
        logging.info('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        """
        Initializes state of the model.

        :param model: model
        """
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer, scheduler, update=True):
        """
        Performs one step of the optimizer.

        :param loss: value of loss function
        :param optimizer: optimizer
        :param update: if True executes weight update
        """
        loss.backward()
        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        if update:
            scheduler.step()
            optimizer.step()
        self.model.zero_grad()
