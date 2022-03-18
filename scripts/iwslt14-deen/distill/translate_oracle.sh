
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat \
  --gen-subset $3 \
  --seed 1234 \
  --task nat \
  --remove-bpe \
  --batch-size 32 \
  --beam 1 \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  -s de -t en \
  --path /home/wangdq/save/inter/iwslt16_deen_distill/$4/checkpoint_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path $2 \
  --model-overrides "{'valid_subset': '$3'}" \
  --left-pad-source False

tail -1 $2/generate-$3.txt