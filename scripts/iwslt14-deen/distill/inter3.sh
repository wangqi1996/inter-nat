export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/
SAVEDIR=/home/wangdq/save/inter/iwslt14_deen_distill/all_inter
LOGDIR=/home/wangdq/log/inter/iwslt14_deen_distill/all_inter
fairseq-train $DISTILL \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat \
  --save-dir $SAVEDIR --tensorboard-logdir $LOGDIR \
  --ddp-backend=no_c10d --task nat --arch inter_iwslt \
  --eval-bleu --eval-bleu-args '{"max_len_a": 1.2, "max_len_b": 10, "max_iter": 0}' \
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
  --lr 3e-4 --lr-scheduler polynomial_decay \
  --end-learning-rate 1e-5 \
  --warmup-updates 0 \
  --total-num-update 250000 \
  --dropout 0.3 \
  --weight-decay 0 \
  --decoder-learned-pos \
  --encoder-learned-pos \
  --apply-bert-init \
  --log-format 'simple' --log-interval 1000 \
  --fixed-validation-seed 7 \
  --max-tokens 4096 \
  --max-update 250000 \
  --num-workers 0 \
  -s de -t en \
  --no-epoch-checkpoints --save-interval-updates 1000 --keep-interval-updates 5 \
  --no-last-checkpoints --validate-after-updates 20000 \
  --keep-best-checkpoints 5 \
  --left-pad-source False \
  --batch-size-valid 20 \
  --weight 2 --dep-file iwslt16_deen_raw
