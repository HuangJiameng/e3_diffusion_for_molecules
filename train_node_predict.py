# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
from absl import logging
import os

# import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save, data_callback
from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask,
)
import qm9.utils as qm9utils
import egnn.node_predict as node_predict
import tqdm

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)


class CFG:
    pass


cfg = CFG()
cfg.batch_size = 128
cfg.num_workers = 0
cfg.filter_n_atoms = None
cfg.datadir = "/sharefs/anonymous/temp"
cfg.model_dir = "/sharefs/anonymous/node_predict/model_ckpt"
cfg.dataset = "qm9"
cfg.remove_h = False
cfg.include_charges = False
cfg.lr = 2e-4
cfg.n_epochs = 200


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def get_optim(model, lr):
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-12
    )

    return optim


def train_epoch(
    data_loader,
    model,
    optimizer,
    device="cuda",
    dtype=torch.float32,
    augment_noise=0,
    data_augmentation=False,
):
    model.train()
    n_iterations = len(data_loader)
    with tqdm.tqdm(total=n_iterations) as pbar:
        correct, total = 0, 0
        mol_correct, mol_total = 0, 0
        for i, data in enumerate(data_loader):
            x = data["positions"].to(device, dtype)
            node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
            edge_mask = data["edge_mask"].to(device, dtype)
            one_hot = data["one_hot"].to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)

            if augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * augment_noise

            x = remove_mean_with_mask(x, node_mask)
            if data_augmentation:
                x = utils.random_rotation(x).detach()

            check_mask_correct([x, one_hot], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {"categorical": one_hot}
            input_h = torch.ones_like(one_hot, dtype=x.dtype)
            input_h = input_h / input_h.shape[-1]
            input_h = input_h.to(device, dtype)
            out_x, out_h = model(0, x, input_h, node_mask, edge_mask)
            # print(x.shape, input_h.shape, out_h.shape, one_hot.shape, node_mask.shape)
            loss = model.loss(out_h, one_hot, node_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.detach().cpu().numpy())
            pbar.update(1)
            _correct = torch.sum(
                torch.logical_and(
                    out_h.argmax(dim=-1) == one_hot.argmax(dim=-1), node_mask.squeeze()
                )
            )
            _mol_correct = torch.sum(
                torch.all(
                    torch.logical_or(
                        out_h.argmax(dim=-1) == one_hot.argmax(dim=-1),
                        torch.logical_not(node_mask.squeeze()),
                    ),
                    dim=-1,
                )
            )
            _mol_total = x.shape[0]
            mol_correct += _mol_correct
            mol_total += _mol_total
            _total = torch.sum(node_mask.squeeze())
            correct += _correct
            total += _total
        print(
            f"per atom train acc: {(correct / total).detach().cpu().numpy()}, per molecule train acc: {(mol_correct / mol_total).detach().cpu().numpy()}"
        )


def eval_epoch(
    data_loader,
    model,
    device="cuda",
    dtype=torch.float32,
    augment_noise=0,
    data_augmentation=False,
):
    model.train()
    n_iterations = len(data_loader)
    with tqdm.tqdm(total=n_iterations) as pbar:
        correct, total = 0, 0
        mol_correct, mol_total = 0, 0
        for i, data in enumerate(data_loader):
            x = data["positions"].to(device, dtype)
            node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
            edge_mask = data["edge_mask"].to(device, dtype)
            one_hot = data["one_hot"].to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)

            if augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * augment_noise

            x = remove_mean_with_mask(x, node_mask)
            if data_augmentation:
                x = utils.random_rotation(x).detach()

            check_mask_correct([x, one_hot], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {"categorical": one_hot}
            input_h = torch.ones_like(one_hot, dtype=x.dtype)
            input_h = input_h / input_h.shape[-1]
            input_h = input_h.to(device, dtype)
            out_x, out_h = model(0, x, input_h, node_mask, edge_mask)
            # print(x.shape, input_h.shape, out_h.shape, one_hot.shape, node_mask.shape)
            loss = model.loss(out_h, one_hot, node_mask)
            pbar.set_postfix(loss=loss.detach().cpu().numpy())
            pbar.update(1)
            _mol_correct = torch.sum(
                torch.all(
                    torch.logical_or(
                        out_h.argmax(dim=-1) == one_hot.argmax(dim=-1),
                        torch.logical_not(node_mask.squeeze()),
                    ),
                    dim=-1,
                )
            )
            _mol_total = x.shape[0]
            mol_correct += _mol_correct
            mol_total += _mol_total
            _correct = torch.sum(
                torch.logical_and(
                    out_h.argmax(dim=-1) == one_hot.argmax(dim=-1), node_mask.squeeze()
                )
            )
            _total = torch.sum(node_mask.squeeze())
            correct += _correct
            total += _total
        print(
            f"Correct: {correct.detach().cpu().numpy()}, Total: {total.detach().cpu().numpy()}, Accuracy: {(correct/total).detach().cpu().numpy()}"
        )
        print(
            f"Molecule correct: {mol_correct.detach().cpu().numpy()}, Total: {mol_total}, Accuracy: {(mol_correct/mol_total).detach().cpu().numpy()}"
        )


def main():
    # create dir if not exists
    os.makedirs(cfg.model_dir, exist_ok=True)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg=cfg)
    dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)
    atom_encoder = dataset_info["atom_encoder"]
    atom_decoder = dataset_info["atom_decoder"]
    in_node_nf = len(dataset_info["atom_decoder"]) + int(cfg.include_charges)
    print(f"in_node_nf: {in_node_nf}")
    model = node_predict.EGNN_dynamics_QM9(
        in_node_nf=in_node_nf, context_node_nf=0, n_dims=3, device="cuda"
    ).cuda()
    optimizer = get_optim(model, lr=cfg.lr)
    for epoch in range(cfg.n_epochs):
        logging.info(f"Epoch {epoch}")
        train_epoch(data_loader=dataloaders["train"], model=model, optimizer=optimizer)

        if epoch % 5 == 0:
            logging.info(f"Eval epoch {epoch} and save model")
            eval_epoch(data_loader=dataloaders["valid"], model=model)
            utils.save_model(model, f"{cfg.model_dir}/model_{epoch}.npy")
            utils.save_model(optimizer, f"{cfg.model_dir}/optim_{epoch}.npy")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
