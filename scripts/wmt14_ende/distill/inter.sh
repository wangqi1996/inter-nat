export CUDA_VISIBLE_DEVICES=$1
export MKL_THREADING_LAYER=GUN
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq/
SAVEDIR=/home/wangdq/save/inter/wmt14_ende_distill/inter_all
LOGDIR=/home/wangdq/log/inter/wmt14_ende_distill/inter_all
rm -rf $LOGDIR
fairseq-train $DISTILL \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat \
  --save-dir $SAVEDIR --tensorboard-logdir $LOGDIR \
  --ddp-backend=no_c10d --task nat --arch inter_wmt \
  --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10, "max_iter": 0}' \
  --eval-bleu-detok moses --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --length-loss-factor 0.1 \
  --criterion nat_base_loss \
  --mapping-func interpolate \
  --mapping-use output \
  --share-all-embeddings \
  --share-rel-embeddings \
  --block-cls highway \
  --self-attn-cls shaw \
  --enc-self-attn-cls shaw \
  --enc-block-cls highway \
  --max-rel-positions 4 \
  --pred-length-offset \
  --noise full_mask \
  --optimizer adam --adam-betas '(0.9,0.999)' \
  --lr 0.0007 --lr-scheduler inverse_sqrt \
  --min-lr 1e-09 --warmup-updates 10000 \
  --warmup-init-lr 1e-07 --clip-norm 5.0 \
  --dropout 0.3 --weight-decay 0.01 --label-smoothing 0.0 \
  --encoder-learned-pos \
  --apply-bert-init \
  --log-format 'simple' --log-interval 1000 \
  --fixed-validation-seed 7 \
  --max-tokens 16000 --update-freq 1 \
  --max-update 300000 \
  -s en -t de \
  --keep-best-checkpoints 5 \
  --no-epoch-checkpoints --validate-after-updates 20000 \
  --save-interval-updates 500 \
  --keep-interval-updates 5 \
  --num-workers 0 \
  --left-pad-source False \
  --batch-size-valid 20 \
  --dep-file wmt14_ende_distill \
  --weight 3 \
  --fp16
#  --restore-file /home/wangdq/save/inter/wmt14_ende_distill/inter/checkpoint_10_20000.pt
