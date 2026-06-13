from torch.utils.data import DataLoader
import loader2 as lo
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
import random
from evaluate_teacher import Evaluate_teacher
from config import *
from Student import student_model as stu_model
from Teacher import teacher_model as tea_model
from utils import sigmoid, maskedNLL, maskedMSE, MSELoss2, CELoss, distillation_loss
import warnings
import math

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    args['train_flag'] = True
    l_path = args['path']
    l_pre_path = args['pre_path']
    name = args['name']
    print(args['path'])

    args['batch_size'] = 128
    print(f"Using batch_size: {args['batch_size']} (required by teacher model)")

    highwayNet_name = '/epoch' + name + '_g.pth'
    highwayNet_path = os.path.join(l_pre_path, highwayNet_name)

    log_var_fut = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_lat = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_lon = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

    log_var = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
    log_var_distill = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

    evaluate_teacher = Evaluate_teacher()

    gdEncoder_stu = stu_model.GDEncoder_stu(args)
    generator_stu = stu_model.Generator_stu(args)

    gdEncoder_stu = gdEncoder_stu.to(device)
    generator_stu = generator_stu.to(device)
    gdEncoder_stu.train()
    generator_stu.train()

    sample_ratio = 0.01

    trSet = lo.ngsimDataset('./data/dataset_t_v_t/TrainSet.mat')
    original_size = len(trSet)
    sampled_size = int(original_size * sample_ratio)

    min_required = args['batch_size'] * 10
    if sampled_size < min_required:
        sampled_size = min(min_required, original_size)
        sample_ratio = sampled_size / original_size
        print(f"Adjusted sample_ratio to {sample_ratio:.4f} to ensure minimum {min_required} samples")

    print(f"Original dataset size: {original_size}, Sampled size: {sampled_size}")

    sampled_indices = random.sample(range(original_size), sampled_size)
    sampled_trSet = torch.utils.data.Subset(trSet, sampled_indices)

    trDataloader = DataLoader(sampled_trSet,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True,
                              collate_fn = trSet.collate_fn)

    effective_batches = len(trDataloader)
    print(f"Total complete batches per epoch: {effective_batches}")

    if effective_batches == 0:
        print("ERROR: No complete batches available! Need to increase sample_ratio or decrease batch_size.")
        return

    params_gdEncoder = list(gdEncoder_stu.parameters()) + [log_var, log_var_distill, log_var_fut, log_var_lat,
                                                           log_var_lon]
    params_generator = list(generator_stu.parameters()) + [log_var, log_var_distill, log_var_fut, log_var_lat,
                                                           log_var_lon]

    optimizer_gd = optim.Adam(params_gdEncoder, lr=learning_rate)
    optimizer_g = optim.Adam(params_generator, lr=learning_rate)

    scheduler_gd = ExponentialLR(optimizer_gd, gamma=0.6)
    scheduler_g = ExponentialLR(optimizer_g, gamma=0.6)

    for epoch in range(args['epoch']):
        print(f"\nEpoch: {epoch + 1}/{args['epoch']}, lr: {optimizer_g.param_groups[0]['lr']:.6f}")

        loss_gi1 = 0
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0
        distil_loss = 0
        distil_loss2 = 0
        distil_loss3 = 0

        successful_batches = 0

        for idx, data in enumerate(tqdm(trDataloader, desc=f"Epoch {epoch + 1}", unit="batch")):
            try:
                hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, \
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, \
                fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data

                actual_batch_size = hist_batch.size(0)

                # 数据移到GPU
                hist_batch = hist_batch.to(device)
                nbrs_batch = nbrs_batch.to(device)
                mask_batch = mask_batch.to(device)
                lat_enc_batch = lat_enc_batch.to(device)
                lon_enc_batch = lon_enc_batch.to(device)
                lane_batch = lane_batch.to(device)
                nbrslane_batch = nbrslane_batch.to(device)
                class_batch = class_batch.to(device)
                nbrsclass_batch = nbrsclass_batch.to(device)
                fut_batch = fut_batch.to(device)
                op_mask_batch = op_mask_batch.to(device)
                va_batch = va_batch.to(device)
                nbrsva_batch = nbrsva_batch.to(device)
                hist_batch_stu = hist_batch_stu.to(device)
                nbrs_batch_stu = nbrs_batch_stu.to(device)
                lane_batch_stu = lane_batch_stu.to(device)
                nbrslane_batch_stu = nbrslane_batch_stu.to(device)
                class_batch_stu = class_batch_stu.to(device)
                nbrsclass_batch_stu = nbrsclass_batch_stu.to(device)
                fut_batch_stu = fut_batch_stu.to(device)
                va_batch_stu = va_batch_stu.to(device)
                nbrsva_batch_stu = nbrsva_batch_stu.to(device)
                edge_index_batch = edge_index_batch.to(device)
                ve_matrix_batch = ve_matrix_batch.to(device)
                ac_matrix_batch = ac_matrix_batch.to(device)
                man_matrix_batch = man_matrix_batch.to(device)
                view_grip_batch = view_grip_batch.to(device)
                graph_matrix = graph_matrix.to(device)

                # 教师网络推理
                with torch.no_grad():
                    fut_pred_tea, lat_pred_tea, lon_pred_tea = evaluate_teacher.main(
                        args['name'], hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                        lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch,
                        edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch,
                        graph_matrix
                    )

                optimizer_g.zero_grad()
                optimizer_gd.zero_grad()

                # 学生网络前向传播
                values = gdEncoder_stu(hist_batch_stu, nbrs_batch_stu, mask_batch, va_batch_stu, nbrsva_batch_stu,
                                       lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu)
                g_out, lat_pred, lon_pred = generator_stu(values, lat_enc_batch, lon_enc_batch)

                # 计算精度参数
                precision = torch.exp(-log_var).to(device)
                precision_distill = torch.exp(-log_var_distill).to(device)
                precision_fut = torch.exp(-log_var_fut).to(device)
                precision_lon = torch.exp(-log_var_lon).to(device)
                precision_lat = torch.exp(-log_var_lat).to(device)

                # 损失计算
                if args.get('use_mse', False):
                    loss_g1 = precision * MSELoss2(g_out, fut_batch_stu,
                                                   op_mask_batch) + precision_distill * distillation_loss(g_out,
                                                                                                          fut_pred_tea)
                else:
                    if epoch < args.get('pre_epoch', 3):
                        loss_g1 = precision * MSELoss2(g_out, fut_batch_stu,
                                                       op_mask_batch) + precision_distill * distillation_loss(g_out,
                                                                                                              fut_pred_tea)
                    else:
                        loss_g1 = precision * maskedNLL(g_out, fut_batch_stu,
                                                        op_mask_batch) + precision_distill * distillation_loss(g_out,
                                                                                                               fut_pred_tea)

                loss_gx_3 = precision * CELoss(lat_pred, lat_enc_batch) + precision_distill * distillation_loss(
                    lat_pred, lat_pred_tea)
                loss_gx_2 = precision * CELoss(lon_pred, lon_enc_batch) + precision_distill * distillation_loss(
                    lon_pred, lon_pred_tea)
                loss_gx = precision_lat * loss_gx_3 + precision_lon * loss_gx_2
                loss_g = precision_fut * loss_g1 + loss_gx + log_var + log_var_distill + log_var_fut + log_var_lat + log_var_lon

                loss_g.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(generator_stu.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(gdEncoder_stu.parameters(), 10)

                optimizer_g.step()
                optimizer_gd.step()

                # 统计损失
                loss_gi1 += loss_g1.item()
                loss_gx_2i += loss_gx_2.item()
                loss_gx_3i += loss_gx_3.item()
                loss_gix += loss_gx.item()
                distil_loss += distillation_loss(g_out, fut_pred_tea).item()
                distil_loss2 += distillation_loss(lat_pred, lat_pred_tea).item()
                distil_loss3 += distillation_loss(lon_pred, lon_pred_tea).item()

                successful_batches += 1

                # 每20个batch输出一次进度
                if successful_batches % 20 == 0:
                    avg_mse = loss_gi1 / successful_batches
                    avg_lat = loss_gx_3i / successful_batches
                    avg_lon = loss_gx_2i / successful_batches
                    avg_distil = distil_loss / successful_batches

                    print(f"\nBatch {successful_batches}:")
                    print(f"  MSE: {avg_mse:.6f} | Lat: {avg_lat:.6f} | Lon: {avg_lon:.6f}")
                    print(f"  Distil: {avg_distil:.6f}")

            except RuntimeError as e:
                error_msg = str(e)
                if "shape" in error_msg and "invalid for input of size" in error_msg:
                    print(f"\nShape error in batch {idx}: {error_msg}")
                    print("This suggests teacher model BS=128 hardcoding issue")
                    continue
                elif "out of memory" in error_msg:
                    print(f"\nOOM error in batch {idx}, try reducing batch_size or sample_ratio")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\nRuntime error in batch {idx}: {error_msg}")
                    raise
            except Exception as e:
                print(f"\nUnexpected error in batch {idx}: {e}")
                continue

        # Epoch总结
        if successful_batches > 0:
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Successful batches: {successful_batches}/{effective_batches}")
            print(f"  Avg MSE Loss: {loss_gi1 / successful_batches:.6f}")
            print(f"  Avg Distillation Loss: {distil_loss / successful_batches:.6f}")
            print(f"  Avg Lateral Loss: {loss_gx_3i / successful_batches:.6f}")
            print(f"  Avg Longitudinal Loss: {loss_gx_2i / successful_batches:.6f}")
            print(f"{'=' * 50}")

            # 保存模型
            save_model(name=str(epoch + 1), gdEncoder=gdEncoder_stu, generator=generator_stu, path=args['path'])
        else:
            print(f"\nEpoch {epoch + 1}: No successful batches! Check data and model compatibility.")

        scheduler_gd.step()
        scheduler_g.step()


def save_model(name, gdEncoder, generator, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    torch.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.pth')
    torch.save(generator.state_dict(), l_path + '/epoch' + name + '_g.pth')
    print(f"✓ Model saved: epoch{name}")


if __name__ == '__main__':
    main()