import logging

import torch
from fairseq import utils
from fairseq.tasks import register_task

from nat_base import NATGenerationTask
from nat_base.generator import reverse

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@register_task('toy_nat')
class ToyGenerationTask(NATGenerationTask):
    """
    Translation (Sequence Generation) task for Non-Autoregressive Transformer
    """

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = reverse(
                self.tgt_dict, toks.int().cpu(), self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, None)
        hyps, refs = [], []

        sentence_correct = 0

        def remove_9(text):
            text = ''.join(text.split(' '))
            index = 0
            while text[index] == '9':
                index += 1

            index2 = len(text) - 1
            while text[index2] == '9':
                index2 -= 1
            return text[index: index2 + 1]

        for i in range(len(gen_out)):
            h = decode(gen_out[i][0]['tokens'])
            r = decode(
                utils.strip_pad(sample['target'][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            hyps.append(h)
            refs.append(r)

            _r = remove_9(r)
            _h = remove_9(h)
            if _r == _h:
                sentence_correct += 1

        sentence_all = len(hyps)

        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none'), sentence_correct, sentence_all
        else:
            return sacrebleu.corpus_bleu(hyps, [refs]), sentence_correct, sentence_all

    def valid_step(self, sample, model, criterion):

        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample['target'])
            loss, sample_size, logging_output = criterion(model, sample)

        if self.args.eval_bleu:
            bleu, sen_correct, sen_all = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]

            logging_output['sen_correct'] = sen_correct
            logging_output['sen_all'] = sen_all
        return loss, sample_size, logging_output
