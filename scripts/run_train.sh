#!/bin/bash
#for run_info in CLASS_ce MULTI_ce_mse_ceo MULTI_ce_mse MULTI_ce_mae MULTI_ce_mae_ceo REGRESS_mae REGRESS_mse
#for run_info in MULTI_ce_mse MULTI_ce_mae REGRESS_mae REGRESS_mse CLASS_FocalLoss MULTI_mtmr
#for run_info in REGRESS_rank_ordinal REGRESS_FocalOrdinalLoss REGRESS_rank_dorn REGRESS_soft_ordinal REGRESS

for run_info in MULTI_ce_mse_ceo MULTI_ce_mse MULTI_ce_mae MULTI_ce_mae_ceo
do
  python train_test_all_cosin_lr_apply_to_cancer.py --run_info ${run_info} --seed 5 --gpu 0,1
done

