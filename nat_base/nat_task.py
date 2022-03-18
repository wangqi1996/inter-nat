import json
import logging
from argparse import Namespace

from fairseq.data import (
    encoders,
)
from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@register_task('nat')
class NATGenerationTask(TranslationLevenshteinTask):
    """
    Translation (Sequence Generation) task for Non-Autoregressive Transformer
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationLevenshteinTask.add_args(parser)
        parser.add_argument('--infer-with-reflen', default=False, action='store_true')
        parser.add_argument('--use-oracle-mat', default=False, action="store_true")
        parser.add_argument('--write-tree', default="", type=str)
        parser.add_argument('--write-reference-pairs', default="", type=str)

    def build_generator(self, models, args, **kwargs):
        # add models input to match the API for SequenceGenerator
        # from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        from .generator import NAGenerator
        return NAGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 0),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            infer_with_reflen=getattr(args, "infer_with_reflen", False),
            args=args
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            gen_args = Namespace(**gen_args)
            gen_args.iter_decode_eos_penalty = getattr(args, 'iter_decode_eos_penalty', 0.0)
            gen_args.iter_decode_max_iter = getattr(args, 'iter_decode_max_iter', 0)
            gen_args.iter_decode_beam = getattr(args, 'iter_decode_with_beam', 1)
            gen_args.iter_decode_external_reranker = getattr(args, 'iter_decode_with_external_reranker', False)
            gen_args.decoding_format = getattr(args, 'decoding_format', None)
            gen_args.iter_decode_force_max_iter = getattr(args, 'iter_decode_force_max_iter', False)
            gen_args.retain_history = getattr(args, 'retain_iter_history', False)
            gen_args.infer_with_tgt = getattr(args, "infer_with_tgt", False)
            gen_args.infer_with_reflen = getattr(args, "infer_with_reflen", False)
            self.sequence_generator = self.build_generator([model], gen_args)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
