export CUDA_VISIBLE_DEVICES=$1
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat3 \
  --gen-subset $3 \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 128 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s de -t en \
  --path /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$2 \
  --model-overrides "{'valid_subset': '$3'}" \
  --left-pad-source False

tail -1 ~/$2/generate-$3.txt

python scripts/average_checkpoints.py --input /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint.best --output /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_ave_best.pt

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat3 \
  --gen-subset $3 \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 32 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s de -t en \
  --path /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_ave_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$2 \
  --model-overrides "{'valid_subset': '$3'}" \
  --left-pad-source False

tail -1 ~/$2/generate-$3.txt
