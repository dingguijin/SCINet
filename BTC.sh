python run_ETTh.py --target close --freq t --data BTC --data_path BTCUSDT.csv --features S  --seq_len 96 --label_len 24 --pred_len 5 --hidden-size 4 --stacks 1 --levels 4 --lr 0.001 --batch_size 8 --dropout 0 --model_name ettm1_S_I96_O24_lr1e-3_bs8_dp0_h4_s1l4 --use_multi_gpu --devices 0,1
