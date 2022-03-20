export CUDA_VISIBLE_DEVICES=$1
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/deen-AT/

#fairseq-generate $DISTILL/ \
#  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat2 \
#  --gen-subset $3 \
#  --seed 1234 \
#  --task nat \
#  --remove-bpe \
#  --batch-size 32 \
#  --beam 1 \
#  --iter-decode-max-iter 0 \
#  --iter-decode-eos-penalty 0 \
#  -s de -t en \
#  --path /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_best.pt \
#  --max-len-a 1.2 \
#  --max-len-b 10 \
#  --results-path ~/$2 \
#  --model-overrides "{'valid_subset': '$3'}" \
#  --left-pad-source False
#
#echo "Bleu: "
#tail -1 ~/$2/generate-$3.txt

TREE=/home/wangdq/predict/tree
rm -rf $TREE
mkdir -p $TREE

fairseq-generate $DISTILL/ \
  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat2 \
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
  --left-pad-source False \
  --infer-with-reflen \
  --write-tree $TREE

echo "Position Accuracy: "
python /home/data_ti5_c/wangdq/new/nat/scripts/iwslt14-deen/distill/accuracy.py $TREE/predict.tree $TREE/reference.tree

#PAIR=/home/wangdq/predict/pair.log
#rm -rf $PAIR
#
#fairseq-generate $DISTILL/ \
#  --user-dir /home/data_ti5_c/wangdq/new/nat/inter_nat2 \
#  --gen-subset $3 \
#  --seed 1234 \
#  --task nat \
#  --batch-size 32 \
#  --beam 1 \
#  --iter-decode-max-iter 0 \
#  --iter-decode-eos-penalty 0 \
#  -s de -t en \
#  --path /home/wangdq/save/inter/iwslt14_deen_distill/$4/checkpoint_best.pt \
#  --max-len-a 1.2 \
#  --max-len-b 10 \
#  --results-path ~/$2 \
#  --model-overrides "{'valid_subset': '$3'}" \
#  --left-pad-source False \
#  --write-reference-pairs $PAIR
#
#grep ^H ~/$2/generate-$3.txt >~/$2/hypo
#echo "token-pair accuracy: "
#python /home/data_ti5_c/wangdq/new/nat/scripts/iwslt14-deen/distill/pair_accuracy.py ~/$2/hypo $PAIR
