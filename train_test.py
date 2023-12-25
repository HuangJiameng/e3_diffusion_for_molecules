import wandb
from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask,
)
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch


def train_epoch(
    args,
    loader,
    epoch,
    deq,
    model,
    model_dp,
    model_ema,
    ema,
    device,
    dtype,
    property_norms,
    optim,
    nodes_dist,
    gradnorm_queue,
    dataset_info,
    prop_dist,
):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data["positions"].to(device, dtype)
        node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
        edge_mask = data["edge_mask"].to(device, dtype)
        one_hot = data["one_hot"].to(device, dtype)
        charges = (data["charges"] if args.include_charges else torch.zeros(0)).to(
            device, dtype
        )

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(
                x.size(), x.device, node_mask
            )
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {"categorical": one_hot, "integer": charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(
                args.conditioning, data, property_norms
            ).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(
            args, deq, model_dp, nodes_dist, x, h, node_mask, edge_mask, context
        )
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.0

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(
                f"Epoch: {epoch}, iter: {i}/{n_iterations}, "
                f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                f"RegTerm: {reg_term.item():.1f}, "
                f"GradNorm: {grad_norm:.1f}",
                end="\r",
            )

        nll_epoch.append(nll.item())
        if (epoch % args.sample_eva_epochs == 0) and (
            i % args.visualize_every_batch == 0
        ):
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(
                    args, device, deq, model_ema, prop_dist, dataset_info, epoch=epoch
                )
            save_and_sample_chain(
                model_ema,
                deq,
                args,
                device,
                dataset_info,
                prop_dist,
                epoch=epoch,
                batch_id=str(i),
            )
            sample_different_sizes_and_save(
                model_ema,
                deq,
                nodes_dist,
                args,
                device,
                dataset_info,
                prop_dist,
                epoch=epoch,
            )
            print(f"Sampling took {time.time() - start:.2f} seconds")

            vis.visualize(
                f"{args.output_dir}/{args.exp_name}/epoch_{epoch}_{i}",
                dataset_info=dataset_info,
                wandb=wandb,
            )
            vis.visualize_chain(
                f"{args.output_dir}/{args.exp_name}/epoch_{epoch}_{i}/chain/",
                dataset_info,
                wandb=wandb,
            )
            if len(args.conditioning) > 0:
                vis.visualize_chain(
                    "%s/%s/epoch_%d/conditional/"
                    % (args.output_dir, args.exp_name, epoch),
                    dataset_info,
                    wandb=wandb,
                    mode="conditional",
                )
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def get_slope(x, h, node_mask, norm_values=[2, 4, 2]):
    """get the slope of the loss function at the current point,return the first self.n_dims, [self.n_dims:-1], [-1]"""

    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

    def normalize(x, h, node_mask):
        x = x / norm_values[0]
        # Casting to float in case h still has long or int type.
        h_cat = (h["categorical"].float()) / norm_values[1] * node_mask
        h_int = (h["integer"].float()) / norm_values[2] * node_mask

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        return x, h

    def sample_combined_position_feature_noise(n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, 3), device=node_mask.device, node_mask=node_mask
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, 6), device=node_mask.device, node_mask=node_mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    x, h = normalize(x, h, node_mask)

    xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

    z = sample_combined_position_feature_noise(xh.size(0), xh.size(1), node_mask)

    u = (1 - 1e-4) * z - xh

    postion_slope = sum_except_batch(u[:, :, :3].square()).mean(0).item()
    categorical_slope = sum_except_batch(u[:, :, 3:-1].square()).mean(0).item()
    integer_slope = sum_except_batch(u[:, :, -1:].square()).mean(0).item()

    return postion_slope, categorical_slope, integer_slope


def data_callback(device, dtype, loader, partition="train"):
    ps_list = []
    cs_list = []
    is_list = []
    max_x = []
    for i, data in enumerate(loader):
        x = data["positions"].to(device, dtype)
        batch_size = x.size(0)
        node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
        edge_mask = data["edge_mask"].to(device, dtype)
        one_hot = data["one_hot"].to(device, dtype)
        charges = (data["charges"]).to(device, dtype)
        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        h = {"categorical": one_hot, "integer": charges}
        p_s, c_s, i_s = get_slope(x, h, node_mask)
        print(
            f"partition: {partition}, batch: {i}, position slope: {p_s}, categorical slope: {c_s}, integer slope: {i_s}, max: {x.abs().max()}"
        )
        ps_list.append(p_s)
        cs_list.append(c_s)
        is_list.append(i_s)
        max_x.append(x.abs().max())
    print(
        f"partition: {partition}, position slope: {np.mean(ps_list)}, categorical slope: {np.mean(cs_list)}, integer slope: {np.mean(is_list)}, max: {np.mean(max_x)}"
    )
    return np.mean(ps_list), np.mean(cs_list), np.mean(is_list)
    # if i % args.n_report_steps == 0:
    #     print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
    #           f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
    #           f"RegTerm: {reg_term.item():.1f}, "
    #           f"GradNorm: {grad_norm:.1f}")


def test(
    args,
    deq,
    loader,
    epoch,
    eval_model,
    device,
    dtype,
    property_norms,
    nodes_dist,
    partition="Test",
):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data["positions"].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
            edge_mask = data["edge_mask"].to(device, dtype)
            one_hot = data["one_hot"].to(device, dtype)
            charges = (data["charges"] if args.include_charges else torch.zeros(0)).to(
                device, dtype
            )

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {"categorical": one_hot, "integer": charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(
                    args.conditioning, data, property_norms
                ).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(
                args, deq, eval_model, nodes_dist, x, h, node_mask, edge_mask, context
            )
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(
                    f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                    f"NLL: {nll_epoch/n_samples:.2f}"
                )

    return nll_epoch / n_samples


def save_and_sample_chain(
    model,
    dequantizer,
    args,
    device,
    dataset_info,
    prop_dist,
    epoch=0,
    id_from=0,
    batch_id="",
):
    one_hot, charges, x = sample_chain(
        args=args,
        dequantizer=dequantizer,
        device=device,
        flow=model,
        n_tries=1,
        dataset_info=dataset_info,
        prop_dist=prop_dist,
    )

    vis.save_xyz_file(
        f"{args.output_dir}/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/",
        one_hot,
        charges,
        x,
        dataset_info,
        id_from,
        name="chain",
    )

    return one_hot, charges, x


def sample_different_sizes_and_save(
    model,
    dequantizer,
    nodes_dist,
    args,
    device,
    dataset_info,
    prop_dist,
    n_samples=5,
    epoch=0,
    batch_size=100,
    batch_id="",
):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args,
            device,
            dequantizer,
            model,
            prop_dist=prop_dist,
            nodesxsample=nodesxsample,
            dataset_info=dataset_info,
        )
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(
            f"{args.output_dir}/{args.exp_name}/epoch_{epoch}_{batch_id}/",
            one_hot,
            charges,
            x,
            dataset_info,
            batch_size * counter,
            name="molecule",
        )


def analyze_and_save(
    epoch,
    model_sample,
    dequantizer,
    nodes_dist,
    args,
    device,
    dataset_info,
    prop_dist,
    n_samples=1000,
    batch_size=100,
):
    print(f"Analyzing molecule stability at epoch {epoch}...")
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {"one_hot": [], "x": [], "node_mask": []}
    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args,
            device,
            dequantizer,
            model_sample,
            dataset_info,
            prop_dist,
            nodesxsample=nodesxsample,
        )

        molecules["one_hot"].append(one_hot.detach().cpu())
        molecules["x"].append(x.detach().cpu())
        molecules["node_mask"].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(
        molecules, dataset_info
    )
    print(validity_dict)
    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log(
            {
                "Validity": rdkit_tuple[0][0],
                "Uniqueness": rdkit_tuple[0][1],
                "Novelty": rdkit_tuple[0][2],
            }
        )
    return validity_dict


def save_and_sample_conditional(
    args, device, deq, model, prop_dist, dataset_info, epoch=0, id_from=0
):
    one_hot, charges, x, node_mask = sample_sweep_conditional(
        args, device, deq, model, dataset_info, prop_dist
    )

    vis.save_xyz_file(
        "%s/%s/epoch_%d/conditional/" % (args.output_dir, args.exp_name, epoch),
        one_hot,
        charges,
        x,
        dataset_info,
        id_from,
        name="conditional",
        node_mask=node_mask,
    )

    return one_hot, charges, x
